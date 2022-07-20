import json
import os

import hydra
import numpy as np
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .utils.utils import get_initialize_config


def estim_act_prob(dist, collection_size, voc_size=30522):
    x = np.zeros(voc_size)
    values = list(dist.values())
    indices = [int(i) for i in dist.keys()]
    x[indices] = values
    return x / collection_size


def create_index_dist(index):
    index_dist = {}
    for k, v in index.index_doc_id.items():
        index_dist[int(k)] = len(v)
    return index_dist


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def flops(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    model = get_model(config, init_dict)

    flops_queries = exp_dict["data"]["flops_queries"]

    out_dim = model.output_dim
    # we estimate the FLOPS from a larger set of queries (the 100k dev queries)
    q_collection = CollectionDatasetPreLoad(data_dir=flops_queries, id_style="row_id")
    q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"],
                                    batch_size=config["index_retrieve_batch_size"],
                                    shuffle=False, num_workers=1)

    print("LOAD MODEL AND DOCUMENT INDEX")
    evaluator = SparseIndexing(model=model, config=config, compute_stats=False, restore=True)
    loaded_model = evaluator.model
    doc_index = evaluator.sparse_index

    print("CREATE QUERY INDEX", flush=True)
    evaluator = SparseIndexing(config=None, model=loaded_model, is_query=True, restore=False, compute_stats=True)
    query_index = evaluator.index(q_loader)["index"]
    lexical_queries_index_dist = create_index_dist(query_index)
    lexical_index_dist = create_index_dist(doc_index)

    p_d = estim_act_prob(lexical_index_dist, collection_size=doc_index.nb_docs(), voc_size=out_dim)
    p_q = estim_act_prob(lexical_queries_index_dist, collection_size=len(q_collection), voc_size=out_dim)
    flops = np.sum(p_d * p_q)
    res = dict(flops=flops)
    out_dir = exp_dict.config.out_dir
    json.dump(res, open(os.path.join(out_dir, "flops.json"), "w"))
    print(res)


if __name__ == "__main__":
    flops()
