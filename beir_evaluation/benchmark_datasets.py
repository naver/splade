#!/usr/bin/env python
# coding: utf-8

import json

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoTokenizer

from models import Splade, BEIRSpladeModel

# set the dir for trained weights
# NOTE: this version only works for max agg in SPLADE, so the two directories below !
# If you want to use old weights ("../weights/flops_best" and "../weights/flops_efficient") for BEIR benchmark,
# change the SPLADE aggregation in SPLADE forward in models.py
model_type_or_dir = "../weights/distilsplade_max"
# model_type_or_dir = "../weights/splade_max"

# loading model and tokenizer:
model = Splade(model_type_or_dir)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
beir_splade = BEIRSpladeModel(model, tokenizer)
all_results = dict()

for dataset in ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact", "trec-covid", "webis-touche2020",
                "climate-fever", "dbpedia-entity", "fever", "hotpotqa", "nq"]:
    print("start:", dataset)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "dataset/{}".format(dataset)
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    dres = DRES(beir_splade)
    retriever = EvaluateRetrieval(dres, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 10, 100, 1000], metric="r_cap")
    res = {"NDCG@10": ndcg["NDCG@10"],
           "Recall@100": recall["Recall@100"],
           "R_cap@100": results2["R_cap@100"]}
    print("res for {}:".format(dataset), res)
    all_results[dataset] = res
json.dump(all_results, open("perf.json", "w"))
