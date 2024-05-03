import os
import shutil
import gzip
import json
from tqdm import tqdm
import argparse
from collections import defaultdict

dataset_tokens={
"nfcorpus":167,
"scifact":160,
"arguana":123,
"scidocs":124,
"fiqa":97,
"quora":15,
"trec-covid":122,
"webis-touche2020":156,
"nq":70,
"dbpedia-entity":49,
"hotpotqa":45,
"climate-fever": 69,
"fever": 69,
"msmarco":50
}

queries_tokens={
"nfcorpus":7,
"scifact":22,
"arguana":128,
"scidocs":15,
"fiqa":15,
"quora":13,
"trec-covid":17,
"webis-touche2020":10,
"nq":12,
"dbpedia-entity":8,
"hotpotqa":20,
"climate-fever": 24,
"fever": 12,
"msmarco":9
}


def sort_prune(x,k):
    return dict(sorted(x.items(), key=lambda item: item[1],reverse=True)[:k])

def prune(dataset):
    k = dataset_tokens[dataset] 
    base_directory = os.path.join("beir",dataset,"docs")
    pruned_directory = os.path.join("beir",dataset,"docs_pruned")
    shutil.rmtree(pruned_directory,ignore_errors=True) 
    os.makedirs(pruned_directory,exist_ok=True)
    for filename in tqdm(os.listdir(base_directory)):
        with gzip.open(os.path.join(base_directory, filename), 'rt') as f:
            with gzip.open(os.path.join(pruned_directory, filename), 'wt') as f2:
                for line in f:
                    _dict = json.loads(line)
                    _dict["content"] = ""
                    _dict["vector"] = sort_prune(_dict["vector"],k)
                    f2.write(json.dumps(_dict)+"\n")

def prune_query(dataset):
    k = queries_tokens[dataset] 
    base_file = os.path.join("beir",dataset,"queries_anserini.tsv")
    pruned_file = os.path.join("beir",dataset,"pruned_anserini.tsv")
    with open(base_file, 'r') as f:
        with open(pruned_file, 'w') as f2:
            for line in f:
                id_, *content = line.split()
                dict_ = defaultdict(int)
                for word in content:
                    dict_[word] += 1
                dict_ = sort_prune(dict_, k)
                all_tokens = ""
                for token, amount in dict_.items():
                    all_tokens += " ".join([token for _ in range(amount)])
                final_string = "{}\t{}\n".format(id_, all_tokens.strip())
                f2.write(final_string)


# Example usage
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="first file path")
args = parser.parse_args()
prune(args.dataset)
prune_query(args.dataset)