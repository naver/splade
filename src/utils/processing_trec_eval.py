import json
import os
from collections import defaultdict

"""bunch of methods to convert from trec_eval format to pytrec_eval (and vice versa) 
"""


def build_json_qrel(qrel_file_path):
    """
    input file has format: 186154  0       1160    1
    """
    temp_d = defaultdict(dict)
    with open(qrel_file_path) as reader:
        for line in reader:
            q_id, _, d_id, rel = line.split("\t")
            temp_d[q_id][d_id] = int(rel)
    print("built qrel file, contains {} queries...", len(temp_d))
    json.dump(dict(temp_d), open(os.path.join(os.path.dirname(qrel_file_path), "qrel.json"), "w"))
