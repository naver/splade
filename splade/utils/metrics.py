from collections import Counter

import pytrec_eval
from pytrec_eval import RelevanceEvaluator


def truncate_run(run, k):
    """truncates run file to only contain top-k results for each query"""
    temp_d = {}
    for q_id in run:
        sorted_run = {k: v for k, v in sorted(run[q_id].items(), key=lambda item: item[1], reverse=True)}
        temp_d[q_id] = {k: sorted_run[k] for k in list(sorted_run.keys())[:k]}
    return temp_d


def mrr_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    mrr = evaluator.evaluate(truncated)
    if agg:
        mrr = sum([d["recip_rank"] for d in mrr.values()]) / max(1, len(mrr))
    return mrr


def evaluate(run, qrel, metric, agg=True, select=None):
    assert metric in pytrec_eval.supported_measures, print("provide valid pytrec_eval metric")
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    res = Counter({})
    if agg:
        for d in out_eval.values():  # when there are several results provided (e.g. several cut values)
            res += Counter(d)
        res = {k: v / len(out_eval) for k, v in res.items()}
        if select is not None:
            string_dict = "{}_{}".format(metric, select)
            if string_dict in res:
                return res[string_dict]
            else:  # If the metric is not on the dict, say that it was 0
                return 0
        else:
            return res
    else:
        return out_eval


def init_eval(metric):
    if metric not in ["MRR@10", "recall@10", "recall@50", "recall@100", "recall@200", "recall@500", "recall@1000"]:
        raise NotImplementedError("provide valid metric")
    if metric == "MRR@10":
        return lambda x, y: mrr_k(x, y, k=10, agg=True)
    else:
        return lambda x, y: evaluate(x, y, metric="recall", agg=True, select=metric.split('@')[1])
