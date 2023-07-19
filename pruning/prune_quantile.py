import json 
import gzip
import os
from tqdm import tqdm
import operator
import argparse
from collections import defaultdict
import array
import numpy as np

def sort_dict_by_value_descending(dict_):
    return sorted(dict_.items(), key=operator.itemgetter(1), reverse=True)

def prune_by_value(dict_, values):
    local_vector = {key: value for key, value in dict_["vector"].items() if value > values[key]}            
    return dict(id=dict_["id"],vector=local_vector)

def main(args):
    print(args,flush=True)
    name = args.name
    default_directory = os.path.join("data",name)
    base_directory = os.path.join("data",name,"base_index")
    type = lambda: array.array("f")
    values = defaultdict(type)
    value_directory = os.path.join(default_directory,"-1_words_{}".format(args.quantile))
    os.makedirs(value_directory,exist_ok=True)

    files = os.listdir(base_directory)
    for idx, filename in enumerate(files):
        print("{}/{}".format(idx+1,len(files)))
        f = os.path.join(base_directory, filename)
        open_fn = gzip.open if ".gz" in filename else open
        type_open = "wt"
        # checking if it is a file
        if os.path.isfile(f):
            with open_fn(f,"r") as reader:
                    for i, line in enumerate(tqdm(reader)):
                        if len(line) > 1:
                            dict_ = json.loads(line)
                            for key, value in dict_["vector"].items():
                                values[key].append(value)
    print("getting quantiles")
    pruning_values = {key: np.quantile(value,args.quantile) for key, value in tqdm(values.items())}
    
    print("saving")

    for idx, filename in enumerate(files):
        print("{}/{}".format(idx+1,len(files)))
        f = os.path.join(base_directory, filename)
        open_fn = gzip.open if ".gz" in filename else open
        type_open = "wt"
        # checking if it is a file
        if os.path.isfile(f):
            file_value_prune = open_fn(os.path.join(value_directory,filename),type_open)

            with open_fn(f,"r") as reader:
                    for i, line in enumerate(tqdm(reader)):
                        if len(line) > 1:
                            dict_ = json.loads(line)
                            prune_dict = prune_by_value(dict_, pruning_values)
                            prune_line = json.dumps(prune_dict) + "\n"
                            file_value_prune.write(prune_line)

            file_value_prune.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--quantile', type=float, default=0., help="quantile to prune")
    args = parser.parse_args()

    main(args)
