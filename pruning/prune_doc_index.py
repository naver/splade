import json 
import gzip
import os
from tqdm import tqdm
import operator
import argparse

def sort_dict_by_value_descending(dict_):
    return sorted(dict_.items(), key=operator.itemgetter(1), reverse=True)


def prune_by_value(dict_, value_to_prune):
    local_vector = {key: value for key, value in dict_["vector"].items() if value > value_to_prune*100}            
    return dict(id=dict_["id"],vector=local_vector)
    

def prune_by_size(dict_, size_to_prune):
    local_vector = {key: value for idx, (key, value) in enumerate(sort_dict_by_value_descending(dict_["vector"])) if idx < size_to_prune}    
    return dict(id=dict_["id"],vector=local_vector)

def main(args):
    print(args,flush=True)
    name = args.name
    default_directory = os.path.join("data",name)
    base_directory = os.path.join("data",name,"base_index")
    value_to_prune = args.value_to_prune
    size_to_prune = args.size_to_prune


    if value_to_prune > 0:
        value_directory = os.path.join(default_directory,"prune_value_{}".format(value_to_prune))
        os.makedirs(value_directory,exist_ok=True)

    if size_to_prune > 0:
        size_directory = os.path.join(default_directory,"prune_size_{}".format(size_to_prune))
        os.makedirs(size_directory,exist_ok=True)
    files = os.listdir(base_directory)
    for idx, filename in enumerate(files):
        print("{}/{}".format(idx+1,len(files)))
        f = os.path.join(base_directory, filename)
        open_fn = gzip.open if ".gz" in filename else open
        type_open = "wt"
        # checking if it is a file
        if os.path.isfile(f):

            if value_to_prune > 0:
                file_value_prune = open_fn(os.path.join(value_directory,filename),type_open)
            if size_to_prune > 0:
                file_size_prune = open_fn(os.path.join(size_directory,filename),type_open)

            with open_fn(f,"r") as reader:
                    for i, line in enumerate(tqdm(reader)):
                        if len(line) > 1:
                            dict_ = json.loads(line)
                            if value_to_prune > 0:
                                prune_dict = prune_by_value(dict_, value_to_prune)
                                prune_line = json.dumps(prune_dict) + "\n"
                                file_value_prune.write(prune_line)

                            if size_to_prune > 0:
                                prune_dict = prune_by_size(dict_, size_to_prune)
                                prune_line = json.dumps(prune_dict) + "\n"
                                file_size_prune.write(prune_line)
            if value_to_prune > 0:
                file_value_prune.close()
            if size_to_prune > 0:
                file_size_prune.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--value_to_prune', type=float, help='Prune by min value', default=0)
    parser.add_argument('--size_to_prune', type=float, help='Prune by max size',default=0)
    args = parser.parse_args()

    main(args)
