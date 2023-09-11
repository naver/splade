import hydra
from omegaconf import DictConfig

from splade.datasets.dataloaders import EvalDataLoader, PairwiseRerankPromptDataloader
from splade.datasets.rerank import EvalDatasetMonoT5, EvalDatasetRerank, EvalDatasetRerankPairwise
from splade.evaluate import evaluate
from splade.models.transformer_rank import TransformerRank, RankT5Encoder
from splade.utils.hydra import hydra_chdir
from splade.tasks.transformer_evaluator import RerankEvaluator, PairwisePromptEvaluator
import os
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.utils.utils import get_initialize_config
from transformers import AutoModelForSeq2SeqLM
import torch
from pathlib import Path
import json
import ir_measures

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
        model = RankT5Encoder(**init_dict)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load(config["checkpoint"],map_location=torch.device("cpu")))
        if torch.cuda.device_count() > 1:
            print(" --- use {} GPUs --- ".format(torch.cuda.device_count()),flush=True)
            model = torch.nn.DataParallel(model)
        model.to(device)

    else:
        model = TransformerRank(**init_dict)
    top_k = config.top_k
    add_prefix = config.get("add_prefix",False)
    folder = exp_dict.data.folder

    print(exp_dict)
    paths = sorted(list(Path(folder).rglob("run.[jJ][sS][oO][nN]")))
    os.environ["IR_DATASETS_HOME"] = exp_dict["ir_datasets"]["dataset_path"]
    import ir_datasets
    datasets = [str(x).replace("/run.json","").replace(folder,"") for x in paths]

    if reranker_type == "PairwisePrompt":
        evaluator = PairwisePromptEvaluator(config=config, model=model,restore=idx==0)
    else:
        evaluator = RerankEvaluator(config=config, model=model, restore=(not reranker_type == "rankT5"))


    for idx, (folder, ir_dataset) in enumerate([(x,y) for x,y in zip(paths,datasets)]):
        folder = str(folder)
#        if "beir" in folder or "car" in folder or "antique" in folder or "dpr" in folder or "gov" in folder:
        if "gov" in folder:
            continue
#        if "climate-fever" not in folder:
#            continue
        #Pretreating data:
        if ir_dataset[0] == "/":
            ir_dataset = ir_dataset[1:]
        name = "{}/{}/{}".format(ir_dataset,exp_dict.data.run_name,top_k)
        
        print("STARTING Dataset {}: {}/{}".format(ir_dataset,idx+1,len(paths)))

        dataset = ir_datasets.load(ir_dataset)

        if reranker_type == "monoT5" or reranker_type == "duoT5":
            data = EvalDatasetMonoT5(run_file=folder, document_dir=dataset, query_dir=dataset, top_k=top_k, finish_qrel=dataset)
            loader = data.iterate()
        elif reranker_type == "PairwisePrompt":
            data = EvalDatasetRerankPairwise(run_file=folder, document_dir=dataset, query_dir=dataset, top_k=top_k, finish_qrel=dataset)
            loader = PairwiseRerankPromptDataloader(dataset=data, batch_size=config["eval_batch_size"],
                                    shuffle=False, num_workers=1,
                                    tokenizer_type=config["tokenizer_type"],
                                    max_length=config["max_length"],
                                    prompt=config["prompt"])
        else:
            force_noload = False
            data = EvalDatasetRerank(run_file=folder, document_dir=dataset, query_dir=dataset, top_k=top_k, finish_qrel=dataset, add_prefix=add_prefix, force_noload=force_noload)
            loader = EvalDataLoader(dataset=data, batch_size=config["eval_batch_size"],
                                    shuffle=False, num_workers=1,
                                    tokenizer_type=config["tokenizer_type"],
                                    max_length=config["max_length"],
                                    return_token_type_ids=config.get("return_token_type_ids",False))
        if reranker_type == "PairwisePrompt":
            evaluator.init_(config=config, dataset_name=name,position_dict=data.initial_position_dicts)
        else:
            evaluator.init_(config=config, dataset_name=name)
        if reranker_type == "monoT5":
            evaluator.evaluate(loader, out_dir=os.path.join(config['out_dir'], name),reranker_type=reranker_type,model_name=exp_dict["init_dict"]["model_type_or_dir"])
        else:
            evaluator.evaluate(loader, out_dir=os.path.join(config['out_dir'], name),reranker_type=reranker_type)
        run = json.load(open(os.path.join(config['out_dir'], name, "run.json")))
        print("CLEANING RUN")
        for query_id, doc_dict in run.items():
            query_dict = dict()
            for doc_id, doc_values in doc_dict.items():
                if query_id != doc_id :
                    query_dict[doc_id] = doc_values
            run[query_id] = query_dict
        json.dump(run, open(os.path.join(config['out_dir'], name, "run.json"),"w"))
        qrel = ir_datasets.load(ir_dataset).qrels_iter()
        perf = ir_measures.calc_aggregate([ir_measures.MRR@10, ir_measures.nDCG@10, ir_measures.nDCG(judged_only=True)@10, ir_measures.Success@5, ir_measures.Judged@10, ir_measures.Recall@1000, ir_measures.Recall@100], qrel, run)
        perf = {str(k).lower():float(v) for k,v in perf.items()}
        perf = {str(k):float(v) for k,v in sorted(perf.items())}
        print(perf)
        json.dump(perf, open(os.path.join(config['out_dir'], name, "perf.json"),"w"))

if __name__ == "__main__":
    test_reranker()
