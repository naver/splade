from typing import Dict
from transformers.trainer import Trainer, logger
from transformers import PreTrainedModel
import torch
import os
import numpy as np


class BaseTrainer(Trainer):

    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    @staticmethod
    def _L1(batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()

    @staticmethod
    def _L0(batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()

    @staticmethod
    def splade_max(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        relu = torch.nn.ReLU(inplace=False)
        values, _ = torch.max(torch.log(1 + relu(output)) * attention_mask.unsqueeze(-1), dim=1)
        return values

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
           model=self.model

        if model.doc_encoder_adapter_name:
            model.doc_encoder.load_adapter(os.path.join(resume_from_checkpoint,model.doc_encoder_adapter_name))
            # query
            if model.query_encoder_adapter_name:
                model.query_encoder.load_adapter(os.path.join(resume_from_checkpoint,'query',model.query_encoder_adapter_name))

        else:
            WEIGHTS_NAME = "pytorch_model.bin"
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            load_result = model.doc_encoder.load_state_dict(state_dict, False)
            #query
            if model.shared_weights:
                model.query_encoder = model.doc_encoder
            else:
                state_dict = torch.load(os.path.join(resume_from_checkpoint, "query",WEIGHTS_NAME), map_location="cpu")
                load_result = model.query_encoder.load_state_dict(state_dict, False)
            # release memory
            del state_dict
            self._issue_warnings_after_load(load_result)


    def _save(self, output_dir, state_dict=None):
        ## SAVE CHECKPOINT !
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            self.model.save(output_dir,self.tokenizer)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)



class IRTrainer(BaseTrainer):

    def __init__(self, n_negatives, shared_weights=True, distillation=False, mse_margin=False, splade_doc=False, dense=False, *args, **kwargs):
        super(IRTrainer, self).__init__(*args, **kwargs)
        self.n_negatives = n_negatives
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.distil_loss = torch.nn.KLDivLoss(reduction="none")
        self.mse_loss = torch.nn.MSELoss(reduction="none")
        self.args.remove_unused_columns = False
        self.lambda_d = self.args.l0d
        self.lambda_q = self.args.l0q
        self.T_d = self.args.T_d
        self.T_q = self.args.T_q
        self.last_celoss = list()
        self.last_distilloss = list()
        self.last_flops = list()
        self.last_anti_zero = list()
        self.last_docs = list()
        self.last_queries = list()
        self.shared_weights = shared_weights
        self.mse_margin = mse_margin
        self.splade_doc = splade_doc
        self.step = 0
        self.dense = dense
        self.distillation=distillation


    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if not self.dense:
            logs["L0_d"] = np.mean(self.last_docs)
            logs["L0_q"] = np.mean(self.last_queries)
            logs["flops_loss"] = np.mean(self.last_flops)
            logs["anti-zero"] = np.mean(self.last_anti_zero)
        logs["ce_loss"] = np.mean(self.last_celoss)
        if self.distillation:
            logs["distil_loss"] = np.mean(self.last_distilloss)

        self.last_docs = list()
        self.last_queries = list()
        self.last_flops = list()
        self.last_anti_zero = list()
        self.last_celoss = list()
        if self.distillation:
            self.last_distilloss = list()

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def compute_lambdas(self):
        self.step += 1
        self.lambda_t_d = min(self.lambda_d, self.lambda_d * ((self.step) / (self.T_d+1)) ** 2)
        self.lambda_t_q = min(self.lambda_q, self.lambda_q * ((self.step) / (self.T_q+1)) ** 2)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        teacher_scores = inputs["scores"]
        del inputs["scores"]

        self.compute_lambdas()
        full_output = model(**inputs)
        queries, docs = full_output # shape (bsz, 1, Vocab), (bsz, nb_neg+1, Vocab)
        scores = torch.bmm(queries,torch.permute(docs,[0,2,1])).squeeze(1) # shape (bsz, nb_neg+1)
        scores_positive = scores[:,:1] # shape (bsz, 1)
        negatives = docs[:,1:,:].reshape(-1,docs.size(2)).T # shape (Vocab, bsz*nb_neg)
        scores_negative = torch.matmul(queries.squeeze(1),negatives) # shape (bsz, bsz*nb_neg)
        all_scores = torch.cat([scores_positive,scores_negative],dim=1) # shape (bsz, bsz*nb_neg+1)
        labels_index = torch.zeros(scores.size(0)).to(scores.device).long() # shape (bsz)
        ce_loss = self.ce_loss(all_scores, labels_index).mean()
        
        # training without distillation; loss= cross-entropy loss
        if self.distillation is False:
            loss=ce_loss
        # distillation with MSE margin loss
        elif self.mse_margin:
            scores_negative_student = scores[:,1:] # shape (bsz, nb_neg)
            scores_positive_student = scores[:,:1] # shape (bsz, 1)
            margin_student = scores_positive_student - scores_negative_student

            teacher_scores = teacher_scores.view(scores.size()).to(scores.device)
            scores_negative_teacher = teacher_scores[:,1:]
            scores_positive_teacher = teacher_scores[:,:1]
            margin_teacher = scores_positive_teacher - scores_negative_teacher

            ranking_loss = self.mse_loss(margin_student,margin_teacher).mean(dim=1).mean(dim=0)
            loss = 0.01*ce_loss + 0.99*ranking_loss
        # distillation with kld loss
        else: 
            temperature = 1
            student_scores = torch.log_softmax(scores*temperature,dim=1)
            teacher_scores = teacher_scores.view(scores.size()).to(scores.device)
            teacher_scores = torch.softmax(teacher_scores*temperature,dim=1)

            ranking_loss = self.distil_loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)
            loss = 0.01*ce_loss + 0.99*ranking_loss

        self.last_celoss.append(ce_loss.cpu().detach().item())
        if self.distillation:
            self.last_distilloss.append(ranking_loss.cpu().detach().item())

        if not self.dense:
            flops = self.lambda_t_d*self._flops(docs.reshape(-1,docs.size(2)))
            if not self.splade_doc:
                flops = flops + self.lambda_t_q*self._L1(queries.squeeze(1))
            anti_zero = 1/(torch.sum(queries)**2) + 1/(torch.sum(docs)**2)
            self.last_docs.append(self._L0(docs.reshape(-1,docs.size(2)).cpu().detach()).item())
            self.last_queries.append(self._L0(queries.reshape(-1,queries.size(2)).cpu().detach()).item())
            self.last_flops.append(flops.cpu().detach().item())
            self.last_anti_zero.append(anti_zero.cpu().detach().item())

            loss = loss + flops + anti_zero

        if not return_outputs:
            return loss
        else:
            return loss, [full_output]

