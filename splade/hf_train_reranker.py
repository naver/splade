import os
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from splade.hf.trainers import RerankerTrainer
from splade.hf.collators import RerankerCollator
from splade.hf.datasets import RerankingDataset
from splade.models.transformer_rank import RankT5EncoderFix
from splade.hf.convertl2i2hf import convert
from splade.utils.utils import get_initialize_config
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME,version_base="1.2")
def hf_train_rerank(exp_dict: DictConfig):

    # mapping yaml/hydra conf into HF data structure
    exp_dict, _, _, _ = get_initialize_config(exp_dict, train=True)
    model_args,data_args,training_args = convert(exp_dict)

    tokenizer = AutoTokenizer.from_pretrained(exp_dict.init_dict.model_type_or_dir)

    if exp_dict.config.reranker_type == "rankT5":
        model = RankT5EncoderFix(exp_dict.init_dict.model_type_or_dir, force_nofp=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(exp_dict.init_dict.model_type_or_dir)

    token_type_ids = False

    data_collator= RerankerCollator(tokenizer=tokenizer,max_length=model_args.max_length, token_type_ids=token_type_ids,
                                    prompt_q=data_args.prompt_q, prompt_d=data_args.prompt_d)

    dataset = RerankingDataset(training_data_type=data_args.training_data_type, # training file type
                            training_file_path=data_args.training_data_path, # path to training file
                            document_dir=data_args.document_dir,             # path to document file (collection)
                            query_dir=data_args.query_dir,                   # path to queri=y file
                            qrels_path=data_args.qrels_path,                 # path to qrels
                            n_negatives=data_args.n_negatives,               # nb negatives in batch
                            nqueries=data_args.n_queries,                    # consider only a subset of <nqueries> queries
                            )

    trainer = RerankerTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset,
        data_collator=data_collator.torch_call,
        tokenizer=tokenizer,
        n_negatives=data_args.n_negatives  # training dataset
    )

    trainer.train()
    final_path = os.path.join(training_args.output_dir,"model")
    os.makedirs(final_path,exist_ok=True)
    trainer.save_model(final_path)
    trainer.save_state()

if __name__ == "__main__":
    hf_train_rerank()
