from torch.utils.data import Dataset
from splade.datasets.datasets import IR_Dataset, IR_Dataset_NoLoad, CollectionDatasetPreLoad
import json
from tqdm.auto import tqdm
from collections import defaultdict
try:
    from pygaggle.rerank.base import Query, Text
except:
    print('warning: could not load pygaggle')


class EvalDatasetRerank(Dataset):
    """
    dataset to use for reranking
    """

    def __init__(self, run_file, document_dir, query_dir, top_k=-1, finish_qrel=None, prompt_q=None, prompt_d=None, force_noload=False):
        if type(query_dir) == str:
            self.query_dataset = CollectionDatasetPreLoad(query_dir,id_style="content_id")
        else:
            self.query_dataset = IR_Dataset(query_dir, information_type="query", sequential_idx=False)
        self.query_list = list()
        print("Finish qrel:", finish_qrel, flush=True)
        if finish_qrel:
            if type(finish_qrel) == str:
                with open(finish_qrel) as reader:
                    all_qrel = json.load(reader)
            else:
                iterator = finish_qrel.qrels_iter()
                all_qrel = defaultdict(dict)
                for x in iterator:
                    all_qrel[str(x.query_id)][str(x.doc_id)] = int(x.relevance)
        print("Loading qrel")
        #TODO unify with the other readers?
        print(run_file)
        all_docs = set()
        if run_file.split(".")[-1] == "trec" or run_file.split(".")[-1] == "txt" or run_file.split(".")[-1] == "tsv":
            with open(run_file) as reader:
                for line in tqdm(reader):
                    qid, _, did, position, score, _ = line.split(" ")
                    if str(qid) in self.query_dataset.data_dict:
                        if not finish_qrel or str(qid) in all_qrel:
                            if top_k <= 0 or int(position) <= top_k:
                                self.query_list.append([qid, did])
                                all_docs.add(did)
        else:
            with open(run_file) as reader:
                result = json.load(reader)
            for query_id, documents in tqdm(result.items()):
                if not finish_qrel or str(query_id) in all_qrel:
                    for idx, (doc_id, score) in enumerate(sorted(documents.items(), reverse=True, key=lambda item: item[1])):
                        if top_k <= 0 or idx < top_k:
                            self.query_list.append([query_id, doc_id])     
                            all_docs.add(doc_id)
        self.prompt_d = prompt_d
        self.prompt_q = prompt_q        
        if type(document_dir) == str:
            self.document_dataset = CollectionDatasetPreLoad(document_dir,id_style="content_id")
        else:
            if force_noload:
                self.document_dataset = IR_Dataset_NoLoad(document_dir)
            else:
                name = ""
                try:
                    name = document_dir._name
                except:
                    pass
                if "fever" not in name:
                    print("all docs")
                    self.document_dataset = IR_Dataset(document_dir, information_type="document", sequential_idx=False, all_docs=all_docs)
                else:
                    print("no all docs")
                    self.document_dataset = IR_Dataset(document_dir, information_type="document", sequential_idx=False)

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        q_id, d_id = self.query_list[idx]
        q = self.query_dataset[q_id][1]
        d_id = d_id.replace('"',"")
        d = self.document_dataset[d_id][1]
        if self.prompt_q:
            q = self.prompt_q.format(q)
        if self.prompt_d:
            d = self.prompt_d.format(d)
            q = "query: {}" + q
            d = " document: {}" + d
        return q_id.strip(), d_id.strip(), q.strip(), d.strip()


class EvalDatasetMonoT5():
    """
    Dataset to integrate with monoT5
    """

    def __init__(self, run_file, document_dir, query_dir, top_k=-1, finish_qrel=None):
        if type(document_dir) == str:
            self.document_dataset = CollectionDatasetPreLoad(document_dir,id_style="content_id")
        else:
            self.document_dataset = IR_Dataset(document_dir, sequential_idx=False)
        
        if type(query_dir) == str:
            self.query_dataset = CollectionDatasetPreLoad(query_dir,id_style="content_id")
        else:
            self.query_dataset = IR_Dataset(query_dir, information_type="query", sequential_idx=False)
        self.query_list_dict = defaultdict(list)
        print("Loading qrel")
        if finish_qrel:
            if type(finish_qrel) == str:
                with open(finish_qrel) as reader:
                    finish_qrel = json.load(reader)
            else:
                iterator = finish_qrel.qrels_iter()
                qrels = defaultdict(dict)
                for x in iterator:
                    qrels[str(x.query_id)][str(x.doc_id)] = int(x.relevance)

        if run_file.split(".")[-1] == "trec" or run_file.split(".")[-1] == "txt" or run_file.split(".")[-1] == "tsv":
            with open(run_file) as reader:
                for line in tqdm(reader):
                    qid, _, did, position, score, _ = line.split(" ")
                    if top_k <= 0 or int(position) <= top_k:
                        try:
                            query = self.query_dataset[qid][1].strip()
                        except:
                            continue
                        if len(self.query_list_dict[qid]) == 0:
                            self.query_list_dict[qid].append(Query(query))
                        document_text = self.document_dataset[did][1].strip()

                        self.query_list_dict[qid].append(Text(document_text, {'docid': did}, 0)) 
                    else:
                        continue
        else:
            with open(run_file) as reader:
                result = json.load(reader)
            for qid, documents in tqdm(result.items()):
                for idx, (did, score) in enumerate(sorted(documents.items(), reverse=True, key=lambda item: item[1])):
                    if top_k <= 0 or idx < top_k:
                        try:
                            query = self.query_dataset[qid][1].strip()
                        except:
                            continue
                        if len(self.query_list_dict[qid]) == 0:
                            self.query_list_dict[qid].append(Query(query))
                        document_text = self.document_dataset[did][1].strip()

                        self.query_list_dict[qid].append(Text(document_text, {'docid': did}, 0)) 
        self.query_list_dict = dict(self.query_list_dict)
        all_queries = list(self.query_list_dict.keys())
        for query in all_queries:
            if finish_qrel:
                if query not in finish_qrel:
                    del[self.query_list_dict[query]]

    def iterate(self):
        return self.query_list_dict.items()

class EvalDatasetRerankPairwise(Dataset):
    """
    dataset to use for reranking
    """

    def __init__(self, run_file, document_dir, query_dir, top_k=-1, finish_qrel=None):
        if type(document_dir) == str:
            self.document_dataset = CollectionDatasetPreLoad(document_dir,id_style="content_id")
        else:
#            self.document_dataset = IR_Dataset_NoLoad(document_dir)
            self.document_dataset = IR_Dataset(document_dir, sequential_idx=False)
        if type(query_dir) == str:
            self.query_dataset = CollectionDatasetPreLoad(query_dir,id_style="content_id")
        else:
            self.query_dataset = IR_Dataset(query_dir, information_type="query", sequential_idx=False)

        self.query_dataset = CollectionDatasetPreLoad(query_dir,id_style="content_id")
        self.pair_list = list()
        print("Finish qrel:", finish_qrel, flush=True)
        if finish_qrel:
            if type(finish_qrel) == str:
                with open(finish_qrel) as reader:
                    all_qrel = json.load(reader)
            else:
                iterator = finish_qrel.qrels_iter()
                all_qrel = defaultdict(dict)
                for x in iterator:
                    all_qrel[str(x.query_id)][str(x.doc_id)] = int(x.relevance)
        print("Loading qrel")
        #TODO unify with the other readers?
        self.doc_dicts = dict()
        self.initial_position_dicts = dict()
        if run_file.split(".")[-1] == "trec" or run_file.split(".")[-1] == "txt" or run_file.split(".")[-1] == "tsv":
            with open(run_file) as reader:
                for line in tqdm(reader):
                    qid, _, did, position, score, _ = line.split(" ")
                    if str(qid) in self.query_dataset.data_dict:
                        if not finish_qrel or str(qid) in all_qrel:
                            if top_k <= 0 or int(position) <= top_k:
                                if qid not in self.doc_dicts:
                                    self.doc_dicts[qid] = list()
                                    self.initial_position_dicts[qid] = dict()
                                self.doc_dicts[qid].append(did)
                                self.initial_position_dicts[qid][did] = 1+int(position)
        else:
            with open(run_file) as reader:
                result = json.load(reader)
            for query_id, documents in tqdm(result.items()):
                if not finish_qrel or str(query_id) in all_qrel:
                    for idx, (doc_id, score) in enumerate(sorted(documents.items(), reverse=True, key=lambda item: item[1])):
                        if top_k <= 0 or idx < top_k:
                            if query_id not in self.doc_dicts:
                                self.doc_dicts[query_id] = list()
                                self.initial_position_dicts[query_id] = dict()
                            self.doc_dicts[query_id].append(doc_id)
                            self.initial_position_dicts[query_id][doc_id] = 1+idx
        for qid, list_did in self.doc_dicts.items():
            for d1 in list_did:
                for d2 in list_did:
                    if d1 != d2:
                        self.pair_list.append((qid, d1, d2))



    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        q_id, passage_a, passage_b = self.pair_list[idx]
        q = self.query_dataset[q_id][1]
        d1 = self.document_dataset[passage_a][1]
        d2 = self.document_dataset[passage_b][1]
        return q_id.strip(), passage_a.strip(), passage_b, q.strip(), d1.strip(), d2.strip()

