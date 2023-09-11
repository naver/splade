from transformers import DefaultDataCollator
import torch

def flatten(xss):
    return [x for xs in xss for x in xs]

class RerankerCollator(DefaultDataCollator):
    def __init__(self, tokenizer, token_type_ids=True,max_length=350, prompt_q=None, prompt_d=None, *args, **kwargs):
        super(RerankerCollator, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_type_ids = token_type_ids
        self.prompt_q = prompt_q
        self.prompt_d = prompt_d

    def torch_call(self, examples):
        queries,docs, scores = zip(*examples)
        docs = flatten(docs)
        queries = flatten(queries)
        if self.prompt_q:
            queries = [self.prompt_q.format(q) for q in queries]
        if self.prompt_d:
            docs = [self.prompt_d.format(d) for d in docs]
        try:
            tokenized = self.tokenizer(queries, docs,
                            add_special_tokens=True,
                            padding="longest",  # pad to max sequence length in batch
                            truncation="only_second",  # truncates to self.max_length
                            max_length=self.max_length,
                            return_attention_mask=True,
                            return_tensors="pt",
                            return_token_type_ids=self.token_type_ids) 
        except:
            tokenized = self.tokenizer(queries, docs,
                            add_special_tokens=True,
                            padding="longest",  # pad to max sequence length in batch
                            truncation="longest_first",  # truncates to self.max_length
                            max_length=self.max_length,
                            return_attention_mask=True,
                            return_tensors="pt",
                            return_token_type_ids=self.token_type_ids) 
        scores = torch.cat(scores,dim=0)
        tokenized["scores"] = scores       
        return tokenized

class L2I_Collator(DefaultDataCollator):
    def __init__(self, tokenizer, max_length=350, *args, **kwargs):
        super(L2I_Collator, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def torch_call(self, examples):
        docs,scores = zip(*examples)
        docs = flatten(docs)
        scores = torch.cat(scores,dim=0)
        tokenized = self.tokenizer(docs,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")       
        tokenized["scores"] = scores
        return tokenized

