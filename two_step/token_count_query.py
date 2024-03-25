import ir_datasets
import os
import shutil
import gzip
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse

def count(dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    lengths = list()
    datasets = ir_datasets.load("beir/{}".format(dataset))
    for query in datasets.queries_iter():    
        local_lengths = tokenizer(query.text).input_ids
        local_lengths = len(set(local_lengths))
        lengths.append(local_lengths)
    print(dataset, np.mean(lengths))

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset name")
args = parser.parse_args()
count(args.dataset)
