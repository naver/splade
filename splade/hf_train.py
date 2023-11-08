
import os 
import json

import hydra
from omegaconf import DictConfig, OmegaConf


from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH

from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import asdict

from splade.hf.trainers import IRTrainer
from splade.hf.collators import L2I_Collator
from splade.hf.datasets import L2I_Dataset, TRIPLET_Dataset
from splade.hf.models import  SPLADE, DPR
from splade.hf.convertl2i2hf import convert
from splade.utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME,version_base="1.2")
def hf_train(exp_dict: DictConfig):

    # mapping yaml/hydra conf into HF data structure
    exp_dict, _, _, _ = get_initialize_config(exp_dict, train=True)
    model_args,data_args,training_args = convert(exp_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    # initialize the model: dense or a splade model (splade-doc,(a)symetric splade etc)
    if model_args.dense:
        model = DPR(
            model_args.model_name_or_path,shared_weights=model_args.shared_weights,n_negatives=data_args.n_negatives,
            tokenizer=tokenizer, model_q=model_args.model_q, pooling=model_args.dense_pooling)

    else:
        model = SPLADE(
            model_args.model_name_or_path,shared_weights=model_args.shared_weights,n_negatives=data_args.n_negatives,
            tokenizer=tokenizer, splade_doc=model_args.splade_doc, model_q=model_args.model_q)
            #adapter_name=model_args.adapter_name, adapter_config=model_args.adapter_config, load_adapter=model_args.load_adapter)

    # load the dataset
    data_collator= L2I_Collator(tokenizer=tokenizer,max_length=model_args.max_length)
    if data_args.training_data_type == 'triplets':
         dataset = TRIPLET_Dataset(data_dir=data_args.training_data_path)
    else:
        dataset = L2I_Dataset(training_data_type=data_args.training_data_type, # training file type
                              training_file_path=data_args.training_data_path, # path to training file
                              document_dir=data_args.document_dir,             # path to document file (collection)
                              query_dir=data_args.query_dir,                   # path to queri=y file
                              qrels_path=data_args.qrels_path,                 # path to qrels
                              n_negatives=data_args.n_negatives,               # nb negatives in batch
                              nqueries=data_args.n_queries,                    # consider only a subset of <nqueries> queries
                              )

    
    trainer = IRTrainer(model=model,                         # the instantiated 🤗 Transformers model to be trained
                        args=training_args,                  # training arguments, defined above
                        train_dataset=dataset,
                        data_collator=data_collator.torch_call,
                        tokenizer=tokenizer,
                        shared_weights=model_args.shared_weights,  # query and document model shared or not
                        splade_doc=model_args.splade_doc,          # model is a spladedoc model
                        n_negatives=data_args.n_negatives,         # nb negatives in batch 
                        dense=model_args.dense)                    # is the model dense or not (DPR or SPLADE)
    
    last_checkpoint = None
    if training_args.resume_from_checkpoint: #os.path.isdir(training_args.output_dir) and  not training_args.overwrite_output_dir:
        last_checkpoint  =  get_last_checkpoint(training_args.output_dir)

    if  trainer.is_world_process_zero():
        print(OmegaConf.to_yaml(exp_dict))


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