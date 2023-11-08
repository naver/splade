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
        print("Preloading dataset", flush=True)
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


class L2I_Dataset(Dataset): 
    """ 
    output : (queries, docs), scores
    """

    def __init__(self, document_dir, query_dir, qrels_path, n_negatives=2, nqueries=-1,training_file_path=None, training_data_type=None):

        print("START DATASET: %s"%document_dir, flush=True)
        self.document_dataset = DatasetPreLoad(document_dir,id_style="content_id")
        self.query_dataset = DatasetPreLoad(query_dir,id_style="content_id")
        
        self.samples = dict()
        if qrels_path is not None:
            print("Loading qrel: %s"%qrels_path, flush=True)
            with open(qrels_path) as reader:
                self.qrels = json.load(reader)
                
            # select a subset of  queries 
            if nqueries > 0:
                from itertools import islice
                self.qrels = dict(islice(self.qrels.items(), nqueries))
        
            ### mapping to str ids   ###
            self.qrels = {str(k):{str(k2):v2 for k2,v2 in v.items()} for k,v in self.qrels.items()  }
            ### filtering non positives
            self.qrels={k:{k2:v2 for k2,v2 in v.items() if int(v2)>=1} for k,v in self.qrels.items() }
        else:
            self.qrels = None

        self.samples = dict()        
        self.n_negatives = n_negatives

        print("READING TRAINING FILE (%s)"%training_data_type, flush=True)
        if training_data_type == 'saved_pkl':
            # output of the "others" (filter already done)
            # data_type: saved_pkl
            # format: [POS:[NEG:score,NEG:score,...]] 
            self.samples = pickle.load(open(training_file_path,"rb"))

        elif training_data_type == 'pkl_dict':
            # a la Nils
            # data_type: pkl_dict
            # cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz: qid/did are int!
            with gzip.open(training_file_path, 'rb') as fIn:
                self.samples = pickle.load(fIn)
                # cast int into str for qids/dids
                # and filter out ig not enough negatives 
                self.samples = {str(k):{str(k2):float(v2) for k2,v2 in v.items()} for k,v in self.samples.items() if ( str(k) in self.qrels and len(v.keys()) >  len(self.qrels[str(k)]) )}

                  
        elif training_data_type == 'trec':
            #data_type:trec
            with open(training_file_path) as reader:
                for line in tqdm(reader):
                    qid, _, did, _, score, _ = line.split(" ")
                    # if query subset used
                    if self.qrels is None  or str(qid) in self.qrels:
                        if str(qid) not in self.samples:
                            self.samples[str(qid)] = dict()
                        self.samples[str(qid)][str(did)]=float(score)

        elif training_data_type == 'json':
            # l2i output
            #data_type: json
            with open(training_file_path) as reader:
                self.result_path = json.load(reader)
            for qid, documents in tqdm(self.result_path.items()):
                # if query subset used
                if  self.qrels is None or str(qid) in self.qrels:
                    if str(qid) not in self.samples:
                        self.samples[str(qid)] = dict()
                    for did, score in sorted(documents.items(), reverse=True, key=lambda item: item[1]):
                        self.samples[str(qid)][str(did)]=float(score)
        else:
            raise NotImplementedError('training_data_type must be in [saved_pkl, pkl_dict, trec, json]')


        self.query_list = list(self.samples.keys())
        print("QUERY SIZE = ", len(self.query_list))
        assert  len(self.query_list) > 0 
       

    def __len__(self):
        return len(self.query_list)


    def __getitem__(self, idx):
        query = self.query_list[idx]
        q = self.query_dataset[query][1]
        positives = list(self.qrels[query].keys())
        
        candidates = [x for x in self.samples[query] if x not in positives]

        positive = random.sample(positives,1)[0]

        if len(candidates) <= self.n_negatives:
            negative_ids = random.choices(candidates,k=self.n_negatives)
        else:
            negative_ids = random.sample(candidates,self.n_negatives)

        d_pos = self.document_dataset[positive][1]
        negatives = [self.document_dataset[negative][1].strip() for negative in negative_ids]
        q = q.strip()
        d_pos = d_pos.strip()
        
        docs = [q,d_pos]
        docs.extend(negatives)
        scores_negatives = [self.samples[query][negative] for negative in negative_ids]
        try: # If there's a score for the positive on the file it uses that score
            scores = [self.samples[query][positive]]
        except KeyError: # else it uses the best score of the negatives.
            scores = [max(v for k,v in self.samples[query].items())]
        scores.extend(scores_negatives)
        scores = torch.tensor(scores)
        scores = scores.view(1,-1)
        return docs, scores
    
class RerankingDataset(L2I_Dataset): 

    def __getitem__(self, idx):
        query = self.query_list[idx]
        q = self.query_dataset[query][1]
        positives = list(self.qrels[query].keys())
        
        candidates = [x for x in self.samples[query] if x not in positives]

        positive = random.sample(positives,1)[0]

        if len(candidates) <= self.n_negatives:
            negative_ids = random.choices(candidates,k=self.n_negatives)
        else:
            negative_ids = random.sample(candidates,self.n_negatives)

        d_pos = self.document_dataset[positive][1]
        negatives = [self.document_dataset[negative][1].strip() for negative in negative_ids]
        q = q.strip()
        d_pos = d_pos.strip()
        
        docs = [d_pos]
        docs.extend(negatives)
        scores_negatives = [self.samples[query][negative] for negative in negative_ids]
        try: # If there's a score for the positive on the file it uses that score
            scores = [self.samples[query][positive]]
        except KeyError: # else it uses the best score of the negatives.
            scores = [max(v for k,v in self.samples[query].items())]
        scores.extend(scores_negatives)
        scores = torch.tensor(scores)
        scores = scores.view(1,-1)
        q = [q for _ in docs]
        return q, docs, scores


class TRIPLET_Dataset():
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("READING TRANING FILE (triplet): %s"%(data_dir), flush=True)
        with open(self.data_dir) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    query, pos, neg = line.split("\t")  # first column is id
                    docs = [query.strip(), pos.strip(), neg.strip()]
                    scores = torch.tensor([0, 0, 0])
                    scores = scores.view(1,-1)
                    self.data_dict[i] = docs, scores
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return self.data_dict[idx]

