
import os 
import json

import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH

from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from .hf.trainers import FirstStageTrainer, DistilTrainer
from .hf.args import ModelArguments, DataTrainingArguments, LocalTrainingArguments
from .hf.collators import L2I_Collator
from .hf.datasets import L2I_Dataset
from .hf.models import  TransformerRep, DenseRep
from dataclasses import asdict


from .hf.args import ModelArguments, DataTrainingArguments, LocalTrainingArguments
from .hf.convertl2i2hf import convert
from .utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def hf_train(exp_dict: DictConfig):

    exp_dict, _, _, _ = get_initialize_config(exp_dict, train=True)
    model_args,data_args,training_args = convert(exp_dict)



    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    if model_args.dense:
        model = DenseRep(
            model_args.model_name_or_path,shared_weights=model_args.shared_weights,n_negatives=data_args.n_negatives,
            tokenizer=tokenizer, model_q=model_args.model_q, pooling=model_args.dense_pooling)

    else:
        model = TransformerRep(
            model_args.model_name_or_path,shared_weights=model_args.shared_weights,n_negatives=data_args.n_negatives,
            tokenizer=tokenizer, splade_doc=model_args.splade_doc, model_q=model_args.model_q,
            adapter_name=model_args.adapter_name, adapter_config=model_args.adapter_config, load_adapter=model_args.load_adapter)

    data_collator= L2I_Collator(tokenizer=tokenizer,max_length=model_args.max_length)

    if data_args.distillation:
        dataset = L2I_Dataset(scores=data_args.scores,
            document_dir=data_args.document_dir,
            query_dir=data_args.query_dir,
            qrels_path=data_args.qrels_path,
            n_negatives=data_args.n_negatives,
            nqueries=data_args.n_queries,
            distil=True)

        trainer = DistilTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=dataset,
            data_collator=data_collator.torch_call,
            tokenizer=tokenizer,
            shared_weights=model_args.shared_weights,
            mse_margin=training_args.mse_margin,
            splade_doc=model_args.splade_doc,
            n_negatives=data_args.n_negatives  
        )         

    else: # firststage
        dataset = L2I_Dataset(scores=data_args.scores,
            document_dir=data_args.document_dir,
            query_dir=data_args.query_dir,
            qrels_path=data_args.qrels_path,
            n_negatives=data_args.n_negatives,
            result_path=data_args.result_path,
            negatives_path=data_args.negatives_path,
            distil=False)

        trainer = FirstStageTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=dataset,
            data_collator=data_collator.torch_call,
            tokenizer=tokenizer,
            shared_weights=model_args.shared_weights,
            splade_doc=model_args.splade_doc,
            n_negatives=data_args.n_negatives,
            dense=model_args.dense  
        )


    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and  not training_args.overwrite_output_dir:
        last_checkpoint  =  get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    final_path = os.path.join(training_args.output_dir,"model")
    os.makedirs(final_path,exist_ok=True)
    trainer.save_model(final_path)

    #trainer.create_model_card()   # need .config

    if  trainer.is_world_process_zero():
        with open(os.path.join(final_path, "model_args.json"), "w") as write_file:
            json.dump(asdict(model_args), write_file, indent=4)
        with open(os.path.join(final_path, "data_args.json"), "w") as write_file:
            json.dump(asdict(data_args), write_file, indent=4)
        with open(os.path.join(final_path, "training_args.json"), "w") as write_file:
            json.dump(training_args.to_dict(), write_file, indent=4)


   


if __name__ == "__main__":
    hf_train()