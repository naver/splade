import argparse
import json

from ..utils.metrics import mrr_k, evaluate


def load_and_evaluate(qrel_file_path, run_file_path, metric):
    with open(qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(run_file_path) as reader:
        run = json.load(reader)
    # for trec, qrel_binary.json should be used for recall etc., qrel.json for NDCG.
    # if qrel.json is used for binary metrics, the binary 'splits' are not correct
    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut")
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10)
        print("MRR@10:", res)
        return {"mrr_10": res}
    else:
        res = evaluate(run, qrel, metric=metric)
        print(metric, "==>", res)
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel_file_path")
    parser.add_argument("--run_file_path")
    parser.add_argument("--metric", default="mrr_10")
    args = parser.parse_args()
    load_and_evaluate(args.qrel_file_path, args.run_file_path, args.metric)
