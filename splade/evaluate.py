import json
import os

import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .evaluation.eval import load_and_evaluate
from .utils.utils import get_dataset_name


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def evaluate(exp_dict: DictConfig):
    eval_qrel_path = exp_dict.data.EVAL_QREL_PATH
    eval_metric = exp_dict.config.eval_metric
    out_dir = exp_dict.config.out_dir

    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics) in enumerate(zip(eval_qrel_path, eval_metric)):
        if qrel_file_path is not None:
            res = {}
            dataset_name = get_dataset_name(qrel_file_path)
            print(eval_metrics)
            for metric in eval_metrics:
                res.update(load_and_evaluate(qrel_file_path=qrel_file_path,
                                             run_file_path=os.path.join(out_dir, dataset_name, "run.json"),
                                             metric=metric))
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res
            json.dump(res, open(os.path.join(out_dir, dataset_name, "perf.json"), "a"))
    json.dump(res_all_datasets, open(os.path.join(out_dir, "perf_all_datasets.json"), "a"))
    return res_all_datasets


if __name__ == "__main__":
    evaluate()
