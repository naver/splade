import os
import shutil
import gzip
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse

def count(dataset):
    lengths = list()
    base_directory = os.path.join("beir",dataset,"docs")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for filename in tqdm(os.listdir(base_directory)):
        all_text = list()
        with gzip.open(os.path.join(base_directory, filename), 'rt') as f:
            for line in f:
                _dict = json.loads(line)
                all_text.append(_dict["content"])
        local_lengths = tokenizer(all_text).input_ids
        local_lengths = [len(set(l)) for l in local_lengths]
        lengths.extend(local_lengths)
    print(dataset, np.mean(lengths))

# Example usage
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset name")
args = parser.parse_args()
count(args.dataset)
