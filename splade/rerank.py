import hydra
from omegaconf import DictConfig

from splade.datasets.dataloaders import EvalDataLoader, PairwiseRerankPromptDataloader
from splade.datasets.rerank import EvalDatasetMonoT5, EvalDatasetRerank, EvalDatasetRerankPairwise
from splade.evaluate import evaluate
from splade.models.transformer_rank import TransformerRank, RankT5Encoder, RankT5EncoderFix
from splade.utils.hydra import hydra_chdir
from splade.tasks.transformer_evaluator import RerankEvaluator, PairwisePromptEvaluator
import os
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.utils.utils import get_initialize_config
from transformers import AutoModelForSeq2SeqLM
import torch


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def test_reranker(exp_dict: DictConfig):
    hydra_chdir(exp_dict)
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict)

    init_dict['fp16'] = True
    reranker_type = config.reranker_type
    
    if reranker_type == "monoT5" or reranker_type == "duoT5":
        model = AutoModelForSeq2SeqLM.from_pretrained(exp_dict["init_dict"]["model_type_or_dir"])
    elif reranker_type == "PairwisePrompt":
        model = AutoModelForSeq2SeqLM.from_pretrained(exp_dict["init_dict"]["model_type_or_dir"],trust_remote_code=True)
    elif reranker_type == "rankT5":
        model = RankT5EncoderFix(**init_dict)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        checkpoint = config["checkpoint"] if "checkpoint" in config else os.path.join(config.checkpoint_dir,"model","pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint,map_location=torch.device("cpu")))
        if torch.cuda.device_count() > 1:
            print(" --- use {} GPUs --- ".format(torch.cuda.device_count()),flush=True)
            model = torch.nn.DataParallel(model)
        model.to(device)

    else:
        model = TransformerRank(**init_dict)
    top_k = config.top_k
    if "hf" in exp_dict:
        try:
            prompt_q = exp_dict.hf.data.get("prompt_q",None)
            prompt_d = exp_dict.hf.data.get("prompt_d",None)
        except:
            print("Has hf but cannot load prompts")
            prompt_q = None
            prompt_d = None

    else:
        prompt_q = None
        prompt_d = None

    print(exp_dict)


    for document_dir, path_run, query_dir, qrel_path, dataset_name, run_name in zip(exp_dict.data.document_dir, exp_dict.data.path_run, exp_dict.data.query_dir, exp_dict.data.EVAL_QREL_PATH, exp_dict.data.dataset_name, exp_dict.data.run_name):

        name = "{}/{}/{}".format(dataset_name,run_name,top_k)
        config["retrieval_name"].append(name)

        if exp_dict.data.docs_ir_dataset:
            os.environ["IR_DATASETS_HOME"] = exp_dict["ir_datasets"]["dataset_path"]
            import ir_datasets

            dataset = ir_datasets.load(document_dir)
        else:
            dataset = document_dir


        if reranker_type == "monoT5" or reranker_type == "duoT5":
            data = EvalDatasetMonoT5(run_file=path_run, document_dir=dataset, query_dir=query_dir, top_k=top_k, finish_qrel=qrel_path)
            loader = data.iterate()
        elif reranker_type == "PairwisePrompt":
            data = EvalDatasetRerankPairwise(run_file=path_run, document_dir=dataset, query_dir=query_dir, top_k=top_k, finish_qrel=qrel_path)
            loader = PairwiseRerankPromptDataloader(dataset=data, batch_size=config["eval_batch_size"],
                                    shuffle=False, num_workers=2,
                                    tokenizer_type=config["tokenizer_type"],
                                    max_length=config["max_length"],
                                    prompt=config["prompt"])
        else:
            data = EvalDatasetRerank(run_file=path_run, document_dir=dataset, query_dir=query_dir, top_k=top_k, finish_qrel=qrel_path, prompt_q=prompt_q, prompt_d=prompt_d, force_noload=True)
            loader = EvalDataLoader(dataset=data, batch_size=config["eval_batch_size"],
                                    shuffle=False, num_workers=2,
                                    tokenizer_type=config["tokenizer_type"],
                                    max_length=config["max_length"],
                                    return_token_type_ids=config.get("return_token_type_ids",False))
        if reranker_type == "PairwisePrompt":
            evaluator = PairwisePromptEvaluator(config=config, model=model, dataset_name=name,position_dict=data.initial_position_dicts)
        else:
            evaluator = RerankEvaluator(config=config, model=model, dataset_name=name, restore=not reranker_type == "rankT5")
        if reranker_type == "monoT5":
            evaluator.evaluate(loader, out_dir=os.path.join(config['out_dir'], name),reranker_type=reranker_type,model_name=exp_dict["init_dict"]["model_type_or_dir"])
        else:
            evaluator.evaluate(loader, out_dir=os.path.join(config['out_dir'], name),reranker_type=reranker_type)

    if exp_dict.data.EVAL_QREL_PATH[0]: 
        # evaluate predictions
        evaluate(exp_dict)


if __name__ == "__main__":
    test_reranker()
