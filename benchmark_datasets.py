#!/usr/bin/env python
# coding: utf-8

# # SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking

# This notebook gives a minimal example usage of our SPLADE model with beir. In a nutshell, SPLADE learns **sparse**, **expansion-based** query/doc representations for efficient first-stage retrieval.
# 
# Sparsity is induced via a regularization applied on representations, whose strength can be adjusted; it is thus possible to control the trade-off between effectiveness and efficiency. For more details, check our paper, and don't hesitate to reach out ! 
# 
# We provide weights for one model (splade learned with distillation)
# 
# | model | MRR@10 (MS MARCO dev) | recall@1000 (MS MARCO dev) | expected FLOPS | ~ avg q length | ~ avg d length | 
# | --- | --- | --- | --- | --- | --- |
# | `splade_distil_v2/` | 36.83 | 97.90 | 3.819 | 25 | 231 |

# In[1]:


from models import Splade, BEIRSpladeModel
from transformers import AutoTokenizer
import json

# In[2]:


# set the dir for trained weights 
model_type_or_dir = "weights/splade_distil_v2"
# model_type_or_dir = "weights/flops_best"


# In[3]:


# loading model and tokenizer

model = Splade(model_type_or_dir)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
beir_splade = BEIRSpladeModel(model,tokenizer)


# In[4]:


from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler

all_results = dict()

for dataset in ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact", "trec-covid", "webis-touche2020", "climate-fever", "dbpedia-entity", "fever", "hotpotqa", "nq"]:
    print(dataset,flush=True)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "dataset".format(dataset)
    data_path = util.download_and_unzip(url, out_dir)


    #### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
    # data folder would contain these files: 
    # (1) nfcorpus/corpus.jsonl  (format: jsonlines)
    # (2) nfcorpus/queries.jsonl (format: jsonlines)
    # (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


    # In[ ]:


    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.evaluation import EvaluateRetrieval

    dres = DRES(beir_splade)

    retriever = EvaluateRetrieval(dres, score_function="dot") # or "dot" for dot-product
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1,10,100,1000]) 
    results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1,10,100,1000], metric = "r_cap")   
    res = {
        "NDCG@10":ndcg["NDCG@10"],
        "Recall@100": recall["Recall@100"],
        "R_cap@100": results2["R_cap@100"]
    }
    print(res,flush=True)
    all_results[dataset] = res
    json.dump(all_results, open("perf.json", "w"))
json.dump(all_results, open("perf.json", "w"))

# In[ ]:




