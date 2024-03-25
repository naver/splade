import os
import shutil
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
import argparse

def reorder(dataset):
    base_file = "pisa-canonical/{}/full/full.documents".format(dataset)
    pruned_file = "pisa-canonical/{}/pruned/pruned.documents".format(dataset)
    reorder_file = "pisa-canonical/{}/reorder".format(dataset)
    original_dict = {}
    new_dict = {}
    with open(base_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            original_dict[idx] = line
    with open(pruned_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            new_dict[line] = idx
    with open(reorder_file,"w") as f:
        for k, v in tqdm(original_dict.items()):
            f.write("{} {}\n".format(k, new_dict[v]))



# Example usage
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="first file path")
args = parser.parse_args()
reorder(args.dataset)
