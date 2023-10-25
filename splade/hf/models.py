import torch
from transformers import AutoModelForMaskedLM, AutoModel
from transformers.trainer import  logger
from transformers import PreTrainedModel
import os
from typing import Dict, List
from splade.utils.utils import generate_bow, clean_bow

# try:
#     from transformers.adapters.configuration import AdapterConfig
#     from transformers.adapters import (
#         HoulsbyConfig,
#         PfeifferConfig,
#         PrefixTuningConfig,
#         LoRAConfig,
#         CompacterConfig
#         )
# except ImportError: print('no adapter version')

class SpladeDoc(torch.nn.Module):

    def __init__(self, tokenizer,output_dim):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.special_tokens_map["pad_token"]
        self.cls_token = self.tokenizer.special_tokens_map["cls_token"]
        self.sep_token = self.tokenizer.special_tokens_map["sep_token"]
        self.mask_token = self.tokenizer.special_tokens_map["mask_token"]
        self.pad_id = self.tokenizer.vocab[self.pad_token]
        self.cls_id = self.tokenizer.vocab[self.cls_token]
        self.sep_id = self.tokenizer.vocab[self.sep_token]
        self.mask_id = self.tokenizer.vocab[self.mask_token]
        self.output_dim = output_dim

    def forward(self, **tokens):
        q_bow = generate_bow(tokens["input_ids"], self.output_dim, device=tokens["input_ids"].device)
        q_bow = clean_bow(q_bow, pad_id = self.pad_id, cls_id=self.cls_id, sep_id=self.sep_id, mask_id=self.mask_id)
        return q_bow

    def _save(self, output_dir, state_dict=None):
        ## SAVE CHECKPOINT !
        pass    

class SPLADE(torch.nn.Module):
    
    @staticmethod
    def splade_max(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        output = output.logits
        relu = torch.nn.ReLU(inplace=False)
        values, _ = torch.max(torch.log(1 + relu(output)) * attention_mask.unsqueeze(-1), dim=1)
        return values

    @staticmethod
    def passthrough(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        return output


    def __init__(self, model_type_or_dir, tokenizer=None, shared_weights=True, n_negatives=-1, splade_doc=False, model_q=None, 
                 #adapter_name: str = None,
                 #adapter_config: str = None, #,Union[str, AdapterConfig] = None, 
                 #load_adapter: str = None,
                 **kwargs):
        """
        output indicates which representation(s) to output ('MLM' for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None
        
        self.shared_weights = shared_weights       
        self.doc_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        
        self.output_dim=self.doc_encoder.config.vocab_size

        self.n_negatives = n_negatives
        self.splade_doc = splade_doc
        self.doc_activation = self.splade_max
        self.query_activation = self.splade_max if not self.splade_doc else self.passthrough

        if splade_doc:
            self.query_encoder = SpladeDoc(tokenizer=tokenizer,output_dim=self.doc_encoder.config.vocab_size)
            #self.query_encoder_adapter_name = adapter_name + "_rep_q" if adapter_name else None
        elif shared_weights:
            self.query_encoder = self.doc_encoder
            #self.query_encoder_adapter_name = None
        else:
            if model_q:
                self.query_encoder = AutoModelForMaskedLM.from_pretrained(model_q)
            else:
                self.query_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
            #self.query_encoder_adapter_name = adapter_name + "_rep_q" if adapter_name else None

        # self.adapter_config = adapter_config
        # self.doc_encoder_adapter_name = adapter_name + "_rep" if adapter_name else None
        # # RV ?? 
        # if load_adapter:
        #     print("Loading adapter {}".format(load_adapter))
        #     self.doc_encoder.load_adapter(load_adapter)
        #     self.doc_encoder.set_active_adapters(self.doc_encoder_adapter_name)
        # elif self.doc_encoder_adapter_name:
        #     self.initialize_adapters()

    # def initialize_adapters(self, **kwargs):
    #         leave_out = [kwargs.get('leave_out', "")] if isinstance(kwargs.get('leave_out', ""), int) \
    #                     else kwargs.get('leave_out', "")
    #         if isinstance(self.adapter_config, str):
    #             if self.adapter_config.lower() == "houlsby":
    #                 config = HoulsbyConfig(leave_out=list(map(int, leave_out.strip().split())))
    #             elif self.adapter_config.lower() == "pfeiffer":
    #                 config = PfeifferConfig(leave_out=list(map(int, leave_out.strip().split())))
    #             elif self.adapter_config.lower() == "prefix_tuning":
    #                 prefix_length = kwargs.get("prefix_length", 30)
    #                 config = PrefixTuningConfig(flat=True, prefix_length=prefix_length)
    #             elif self.adapter_config.lower() == "lora":
    #                 r = kwargs.get("r", 8)
    #                 alpha = kwargs.get("alpha", 16)
    #                 config = LoRAConfig(r=r, alpha=alpha)
    #             elif self.adapter_config == "compacter":
    #                 config = CompacterConfig()
    #             else:
    #                 raise ValueError('Adapter Config can be of type: 1. houlsby\n 2. pfeiffer\n 3.prefix_tuning \n4. lora\n')
    #         elif isinstance(self.adapter_config, AdapterConfig):
    #             config = self.adapter_config
    #         else:
    #             original_ln_after = kwargs.pop("original_ln_after", True)
    #             residual_before_ln = kwargs.pop("residual_before_ln", True)
    #             adapter_residual_before_ln = kwargs.pop("adapter_residual_before_ln", True)
    #             ln_before = kwargs.pop("ln_before", True)
    #             ln_after = kwargs.pop("ln_after", True)
    #             mh_adapter = kwargs.pop("mh_adapter", True)
    #             output_adapter = kwargs.pop("output_adapter", True)
    #             non_linearity = kwargs.pop("non_linearity", "relu")
    #             reduction_factor = kwargs.pop("reduction_factor", 64)
    #             inv_adapter = kwargs.pop("inv_adapter", None)
    #             inv_adapter_reduction_factor = kwargs.pop("inv_adapter_reduction_factor", 64)
    #             cross_adapter = kwargs.pop("cross_adapter", True)
    #             config = AdapterConfig(original_ln_after=original_ln_after,
    #                                    residual_before_ln=residual_before_ln,
    #                                    adapter_residual_before_ln=adapter_residual_before_ln,
    #                                    ln_before=ln_before,
    #                                    ln_after=ln_after,
    #                                    mh_adapter=mh_adapter,
    #                                    output_adapter=output_adapter,
    #                                    non_linearity=non_linearity,
    #                                    reduction_factor=reduction_factor,
    #                                    inv_adapter=inv_adapter,
    #                                    inv_adapter_reduction_factor=inv_adapter_reduction_factor,
    #                                    cross_adapter=cross_adapter,
    #                                    leave_out=leave_out
    #                                    )
    #         # load adapters from local directory for resuming training or evaluation
    #         if os.path.isdir(self.doc_encoder_adapter_name): 
    #             self.doc_encoder.load_adapter(self.doc_encoder_adapter_name)
    #             if self.query_encoder_adapter_name and os.path.isdir(self.query_encoder_adapter_name): 
    #                 self.query_encoder.load_adapter(self.query_encoder_adapter_name)
    #         else: # add new adapters for training from scratch
    #             self.doc_encoder.add_adapter(self.doc_encoder_adapter_name, config=config)
    #             if self.query_encoder_adapter_name:
    #                 self.query_encoder.add_adapter(self.query_encoder_adapter_name, config=config)
    #         self.doc_encoder.set_active_adapters(self.doc_encoder_adapter_name)
    #         self.doc_encoder.train_adapter(self.doc_encoder_adapter_name)
    #         if self.query_encoder_adapter_name:
    #             self.query_encoder.set_active_adapters(self.query_encoder_adapter_name)
    #             self.query_encoder.train_adapter(self.query_encoder_adapter_name)


    def forward(self, **tokens):

        if not self.shared_weights or self.splade_doc:
            attention_mask = tokens["attention_mask"]
            input_ids = tokens["input_ids"] ##(bsz * (nb_neg+2) , seq_length)
            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1)) ##(bsz, nb_neg+2 , seq_length)
            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2)) ##(bsz * (nb_neg+1) , seq_length)
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))  ##(bsz * (1) , seq_length)
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            queries_result = self.query_activation(self.query_encoder(input_ids=queries_ids,attention_mask=queries_attention), attention_mask=queries_attention)
            queries_result = queries_result.view(-1,1,queries_result.size(1))  ##(bsz, (1) , Vocab)
            docs_result = self.doc_activation(self.doc_encoder(input_ids=docs_ids,attention_mask=docs_attention),attention_mask=docs_attention)
            docs_result = docs_result.view(-1,self.n_negatives+1,docs_result.size(1))  ####(bsz, (nb_neg+1) , Vocab)
        else:
            representations = self.doc_activation(self.doc_encoder(**tokens),attention_mask=tokens["attention_mask"]) #TODO This should separate docs and queries and use their separate activations, for now is not a problem because they will always be the same if we are here.
            output = representations.view(-1,self.n_negatives+2,representations.size(1))
            queries_result = output[:,:1,:]
            docs_result = output[:,1:,:]
        return queries_result,docs_result

    def save(self,output_dir, tokenizer):
        # if self.doc_encoder_adapter_name and self.doc_encoder.active_adapters:
        #     self.doc_encoder.save_all_adapters(output_dir)
        # else:
        model_dict = self.doc_encoder.state_dict()
        torch.save(model_dict, os.path.join(output_dir,  "pytorch_model.bin"))
        self.doc_encoder.config.save_pretrained(output_dir)

        if not self.shared_weights:
            query_output_dir = os.path.join(output_dir,"query")
            os.makedirs(query_output_dir, exist_ok=True)
            # if self.doc_encoder_adapter_name and self.query_encoder.active_adapters:
                # self.query_encoder.save_all_adapters(query_output_dir)
            # else:
            self.query_encoder.save_pretrained(query_output_dir)
            self.query_encoder.config.save_pretrained(query_output_dir)
            if tokenizer:
                tokenizer.save_pretrained(query_output_dir)

        if tokenizer:
            tokenizer.save_pretrained(output_dir)


class DPR(torch.nn.Module):

    def __init__(self, model_type_or_dir, shared_weights=True, n_negatives=-1, tokenizer=None, model_q=None, pooling='cls'):
        """
        output indicates which representation(s) to output ('MLM' for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        self.shared_weights = shared_weights       
        self.doc_encoder = AutoModel.from_pretrained(model_type_or_dir)
        self.n_negatives = n_negatives
        self.tokenizer = tokenizer
        self.pooling = pooling
        if shared_weights:
            self.query_encoder = self.doc_encoder
        else:
            if model_q:
                self.query_encoder = AutoModel.from_pretrained(model_q)
            else:
                self.query_encoder = AutoModel.from_pretrained(model_type_or_dir)

    @staticmethod
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def forward(self, **tokens):
        if not self.shared_weights:
            attention_mask = tokens["attention_mask"]
            input_ids = tokens["input_ids"]
            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1))
            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2))
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            query_result = self.query_encoder(input_ids=queries_ids,attention_mask=queries_attention)
            query_result = query_result[0]
            if self.pooling == 'mean':
                queries_result = self.mean_pooling(query_result, queries_attention)
            elif  self.pooling == 'cls': 
                queries_result = query_result[:,0,:]

            queries_result = queries_result.view(-1,1,queries_result.size(1))

            docs_result = self.doc_encoder(input_ids=docs_ids,attention_mask=docs_attention)[0]
            if self.pooling == 'mean':
                docs_result = self.mean_pooling(docs_result, queries_attention)
            else:
                docs_result = docs_result[:,0,:]
            docs_result = docs_result.view(-1,self.n_negatives+1,docs_result.size(1))
        else:
            output = self.doc_encoder(**tokens)[0]
            if self.pooling == 'mean':
                output = self.mean_pooling(output, tokens["attention_mask"])
            else:
                output = output[:,0,:]
            output = output.view(-1,self.n_negatives+2,output.size(1))
            queries_result = output[:,:1,:]
            docs_result = output[:,1:,:]
        return queries_result,docs_result

            

    def save(self,output_dir, tokenizer):
        self.doc_encoder.save_pretrained(output_dir)
        if not self.shared_weights:
            query_output_dir = os.path.join(output_dir,"query")
            os.makedirs(query_output_dir, exist_ok=True)
            self.query_encoder.save_pretrained(query_output_dir)
            if tokenizer:
                tokenizer.save_pretrained(query_output_dir)
