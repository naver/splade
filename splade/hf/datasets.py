import json
import gzip
import pickle
import os
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random


class DatasetPreLoad():
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    """

    def __init__(self, data_dir, id_style):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        self.id_style = id_style

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        self.line_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("Preloading dataset %s"%data_dir, flush=True)
        with open(self.data_dir) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    id_ = id_.strip()
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
                        self.line_dict[id_] = i

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex


    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]



class RerankerDataset(Dataset):
    """
    For Reranker only
    output: queries, docs
    """


    def __init__(self, document_dir, query_dir, qrels_path, top_k=-1, n_negatives=2, negatives=None, result_path=None, scores=None, margin=0.15):
        print("START DATASET", flush=True)
        self.document_dataset = DatasetPreLoad(document_dir,id_style="content_id")
        self.query_dataset = DatasetPreLoad(query_dir,id_style="content_id")
        print("Loading qrel")
        with open(qrels_path) as reader:
            self.qrels = json.load(reader)

        if not negatives:
            self.negatives = dict()
            count = 0
            if scores:
                print("READING NILS FILE", flush=True)
                with gzip.open(scores, 'rb') as fIn:
                    scores_dict = pickle.load(fIn)
                for qid, docs in tqdm(scores_dict.items()):
                    if str(qid) in self.qrels:
                        positive_score = -999
                        for positive_str,score in self.qrels[str(qid)].items():
                            if positive_score == -999:
                                positive_score = score
                            else:
                                positive_score = min(positive_score,score)
                        count += 1
                        if str(qid) not in self.negatives:
                            self.negatives[str(qid)] = dict()
                        for did, score in docs.items():
                            if str(did) not in self.qrels[str(qid)]:
                                if True or positive_score - score > margin:
                                    self.negatives[str(qid)][str(did)] = 0
                    if len(self.negatives[str(qid)]) <= n_negatives:
                        del self.negatives[str(qid)]


            if result_path:
                print("READING SPLADE FILE", flush=True)
                if result_path.split(".")[-1] == "trec" or result_path.split(".")[-1] == "txt":
                    with open(result_path) as reader:
                        for line in tqdm(reader):
                            qid, _, did, position, score, _ = line.split(" ")
                            if str(qid) in self.qrels:
                                if str(qid) not in self.negatives:
                                    self.negatives[str(qid)] = dict()
                                if top_k <= 0 or int(position) <= top_k:
                                    if did not in self.qrels[qid]:
                                        self.negatives[qid][did]=0
                else:
                    with open(result_path) as reader:
                        self.result_path = json.load(reader)
                    for qid, documents in tqdm(self.result_path.items()):
                        if str(qid) in self.qrels:
                            if str(qid) not in self.negatives:
                                self.negatives[str(qid)] = dict()
                            for idx, (did, score) in enumerate(sorted(documents.items(), reverse=True, key=lambda item: item[1])):
                                if top_k <= 0 or idx < top_k:
                                    if did not in self.qrels[qid]:
                                        self.negatives[qid][did]=0
        else:
            print("Loading negatives",flush=True)
            with open(negatives,"rb") as f:
                self.negatives = pickle.load(f)

        self.query_list = list(self.negatives.keys())
        print("QUERY SIZE = ", len(self.query_list))
        self.convert = int #if not is_ir else str
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        query = self.query_list[idx]
        q = self.query_dataset[str(query)][1]
        positives = list(self.qrels[str(query)].keys())
        candidates = list(self.negatives[query].keys())
        positive = random.sample(positives,1)[0]
        if len(candidates) <= self.n_negatives:
            negative_ids = random.choices(candidates,k=self.n_negatives)
        else:
            negative_ids = random.sample(candidates,self.n_negatives)

        d_pos = self.document_dataset[positive][1]
        negatives = [self.document_dataset[str(negative)][1].strip() for negative in negative_ids]
        q = q.strip()
        positive = d_pos.strip()
        docs = [positive]
        docs.extend(negatives)
        queries = [q for _ in docs]
        return queries, docs

class L2I_Dataset(Dataset): 
    """ 
    output : (queries, docs), scores
    """

    def __init__(self, document_dir, query_dir, qrels_path, n_negatives=2, nqueries=-1,scores=None, result_path=None, negatives_path=None,distil=True):
        print(f"START LOADING DATASET ({document_dir},{query_dir})", flush=True)
        self.document_dataset = DatasetPreLoad(document_dir,id_style="content_id")
        self.query_dataset = DatasetPreLoad(query_dir,id_style="content_id")
        print(f"Loading qrel ({qrels_path})")
        with open(qrels_path) as reader:
            self.qrels = json.load(reader)
        if nqueries > 0:
            from itertools import islice
            self.qrels = dict(islice(self.qrels.items(), nqueries))
        self.negatives = dict()
        count = 0
        self.distil = distil
        self.convert = int #if not is_ir else str
        self.n_negatives = n_negatives

        if negatives_path:
            # format: [POS:[NEG,NEG,...]] no scoress
            self.negatives = pickle.load(open(negatives_path,"rb"))
        else:
            if scores:
                print("READING SCORES FILE", flush=True)
                with gzip.open(scores, 'rb') as fIn:
                    self.scores_dict = pickle.load(fIn)
                for qid, docs in tqdm(self.scores_dict.items()):
                    if str(qid) in self.qrels:
                        count += 1
                        if str(qid) not in self.negatives:
                            self.negatives[str(qid)] = dict()
                        for did, score in docs.items():
                            if str(did) not in self.qrels[str(qid)]:
                                self.negatives[str(qid)][str(did)] = score
            else:
                if result_path:
                    print("READING SPLADE FILE", flush=True)
                    if result_path.split(".")[-1] == "trec" or result_path.split(".")[-1] == "txt":
                        with open(result_path) as reader:
                            for line in tqdm(reader):
                                qid, _, did, position, score, _ = line.split(" ")
                                if str(qid) in self.qrels:
                                    if str(qid) not in self.negatives:
                                        self.negatives[str(qid)] = dict()
                                    if str(did) not in self.qrels[str(qid)]:
                                        self.negatives[str(qid)][str(did)]=score
                    else:
                        with open(result_path) as reader:
                            self.result_path = json.load(reader)
                        print("SIZE result", len(self.result_path))
                        for qid, documents in tqdm(self.result_path.items()):
                            if str(qid) in self.qrels:
                                if str(qid) not in self.negatives:
                                    self.negatives[str(qid)] = dict()
                                for idx, (did, score) in enumerate(sorted(documents.items(), reverse=True, key=lambda item: item[1])):
                                    if str(did) not in self.qrels[qid]:
                                        self.negatives[str(qid)][str(did)]=score

        self.query_list = list(self.negatives.keys())
        print("QUERY SIZE = ", len(self.query_list))
       

    def __len__(self):
        return len(self.query_list)


    def __getitem__(self, idx):
        query = self.query_list[idx]
        q = self.query_dataset[str(query)][1]
        positives = list(self.qrels[str(query)].keys())
        candidates = list(self.negatives[query].keys())
        positive = random.sample(positives,1)[0]
        if len(candidates) <= self.n_negatives:
            negative_ids = random.choices(candidates,k=self.n_negatives)
        else:
            negative_ids = random.sample(candidates,self.n_negatives)
        d_pos = self.document_dataset[positive][1]
        negatives = [self.document_dataset[str(negative)][1].strip() for negative in negative_ids]
        q = q.strip()
        d_pos = d_pos.strip()
        
        docs = [q,d_pos]
        docs.extend(negatives)
        if self.distil:
            scores = [self.scores_dict[int(query)][int(positive)]]
            scores_negatives = [self.scores_dict[int(query)][int(negative)] for negative in negative_ids]
        else:
            scores = [0]
            scores_negatives = [self.negatives[str(query)][str(negative)] for negative in negative_ids]

        scores.extend(scores_negatives)
        scores = torch.tensor(scores)
        scores = scores.view(1,-1)
        return docs, scores


