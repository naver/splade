import json
import os
from collections import defaultdict

import torch
from omegaconf import open_dict
from tqdm.auto import tqdm

from transformers import PreTrainedModel
from transformers.modeling_utils import unwrap_model
from transformers.adapters.composition import AdapterCompositionBlock, Fuse
from transformers.adapters.utils import WEIGHTS_NAME, CONFIG_NAME

from src.tasks import amp
from src.tasks.base.trainer import TrainerIter
from src.utils.metrics import init_eval
from src.utils.utils import parse
from src.utils.utils import remove_old_ckpt, remove_old_ckpt_adapters


class TransformerTrainer(TrainerIter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.getsize(os.path.join(self.checkpoint_dir, "training_perf.txt")) == 0:
            self.training_res_handler.write("iter,batch_ranking_loss\n")
        if self.validation:
            to_write = "iter"
            if self.validation_loss_loader is not None:
                to_write += ",val_ranking_loss"
            if self.validation_evaluator is not None:
                assert "validation_metrics" in self.config, "need to provide validation metrics"
                self.validation_metrics = self.config["validation_metrics"]
                to_write += ",{}".format(
                    ",".join(["full_rank_{}".format(metric) for metric in self.config["validation_metrics"]]))
                assert "val_full_rank_qrel_path" in self.config, "need to provide path for qrel with this loader"
                self.full_rank_qrel = json.load(open(self.config["val_full_rank_qrel_path"]))
            if os.path.getsize(os.path.join(self.checkpoint_dir, "validation_perf.txt")) == 0:  # if not resuming
                self.validation_res_handler.write(to_write + "\n")
        if self.test_loader is not None:
            pass
        assert "gradient_accumulation_steps" in self.config, "need to setup gradient accumulation steps in config"

    def forward(self, batch):
        """method that encapsulates the behaviour of a trainer 'forward'"""
        raise NotImplementedError

    def evaluate_loss(self, data_loader):
        raise NotImplementedError

    def evaluate_full_ranking(self, i):
        raise NotImplementedError

    def train_iterations(self):
        moving_avg_ranking_loss = 0
        mpm = amp.MixedPrecisionManager(self.fp16)
        self.optimizer.zero_grad()

        for i in tqdm(range(self.start_iteration, self.nb_iterations + 1)):
            self.model.train()  # train model
            # self.optimizer.zero_grad()
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                # when nb_iterations > len(data_loader)
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)

            with mpm.context():
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                out = self.forward(batch)  # out is a dict (we just feed it to the loss)
                loss = self.loss(out).mean()  # we need to average as we obtain one loss per GPU in DataParallel
                moving_avg_ranking_loss = 0.99 * moving_avg_ranking_loss + 0.01 * loss.item()
                # training moving average for logging
                if self.regularizer is not None:
                    if "train" in self.regularizer:
                        regularization_losses = {}
                        for reg in self.regularizer["train"]:
                            lambda_q = self.regularizer["train"][reg]["lambdas"]["lambda_q"].step() if "lambda_q" in \
                                                                                                       self.regularizer[
                                                                                                           "train"][
                                                                                                           reg][
                                                                                                           "lambdas"] else False
                            lambda_d = self.regularizer["train"][reg]["lambdas"]["lambda_d"].step() if "lambda_d" in \
                                                                                                       self.regularizer[
                                                                                                           "train"][
                                                                                                           reg][
                                                                                                           "lambdas"] else False
                            targeted_rep = self.regularizer["train"][reg]["targeted_rep"]  # used to select the "name"
                            # of the representation to regularize (for instance the model could output several
                            # representations e.g. a semantic rep and a lexical rep) => this is just a general case
                            # for the Trainer
                            regularization_losses[reg] = 0
                            if lambda_q:
                                regularization_losses[reg] += (self.regularizer["train"][reg]["loss"](
                                    out["pos_q_{}".format(targeted_rep)]) * lambda_q).mean()
                            if lambda_d:
                                regularization_losses[reg] += ((self.regularizer["train"][reg]["loss"](
                                    out["pos_d_{}".format(targeted_rep)]) * lambda_d).mean() +
                                                               (self.regularizer["train"][reg]["loss"](
                                                                   out["neg_d_{}".format(
                                                                       targeted_rep)]) * lambda_d).mean()) / 2
                            # NOTE: we take the rep of pos q for queries, but it would be equivalent to take the neg
                            # (because we consider triplets, so the rep of pos and neg are the same)
                            loss += sum(regularization_losses.values())
                    with torch.no_grad():
                        monitor_losses = {}
                        for reg in self.regularizer["eval"]:
                            monitor_losses["{}_q".format(reg)] = self.regularizer["eval"][reg]["loss"](
                                out["pos_q_rep"]).mean()
                            # again, we can choose pos_q_rep or neg_q_rep indifferently
                            monitor_losses["{}_d".format(reg)] = (self.regularizer["eval"][reg]["loss"](
                                out["pos_d_rep"]).mean() + self.regularizer["eval"][reg]["loss"](
                                out["neg_d_rep"]).mean()) / 2
            # when multiple GPUs, we need to aggregate the loss from the different GPUs (that's why the .mean())
            # see https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
            # for gradient accumulation  # TODO: check if everything works with gradient accumulation
            loss = loss / self.config["gradient_accumulation_steps"]
            # perform gradient update:
            mpm.backward(loss)
            if i % self.config["gradient_accumulation_steps"] == 0:
                mpm.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], i - 1)
            if i % self.config["train_monitoring_freq"] == 0:
                self.training_res_handler.write("{},{}\n".format(i, loss.item()))
                self.writer.add_scalar("batch_train_loss", loss.item(), i)
                self.writer.add_scalar("moving_avg_ranking_loss", moving_avg_ranking_loss, i)
                print("+batch_loss_iter{}: {}".format(i, round(loss.item(), 4)))
                if self.regularizer is not None:
                    if "train" in self.regularizer:
                        for reg_loss in regularization_losses:
                            self.writer.add_scalar("batch_{}".format(reg_loss),
                                                   regularization_losses[reg_loss].item(), i)
                    for monitor_loss in monitor_losses:
                        self.writer.add_scalar("batch_{}".format(monitor_loss),
                                               monitor_losses[monitor_loss].item(), i)
            # various metrics we save:
            if i % self.record_frequency == 0:
                if self.validation or (self.test_loader is not None):
                    self.model.eval()
                    with torch.no_grad():
                        if self.validation:
                            self.validation_res_handler.write("{}".format(i))
                            if self.validation_loss_loader is not None:
                                val_res_loss = self.evaluate_loss(data_loader=self.validation_loss_loader)
                                self.validation_res_handler.write(",{}".format(val_res_loss["val_ranking_loss"]))
                                for k, v in val_res_loss.items():
                                    self.writer.add_scalar(k, val_res_loss[k], i)
                                if self.val_decision == "loss":
                                    val_dec = val_res_loss["val_ranking_loss"]
                                    print("~~VAL_RANKING_LOSS_iter{}: {}".format(i, val_dec, 4))
                            if self.validation_evaluator is not None:
                                val_res_ranking = self.evaluate_full_ranking(i)
                                self.validation_res_handler.write(",{}".format(",".join(
                                    [str(val_res_ranking["val_FULL_{}".format(metric)]) for metric in
                                     self.config["validation_metrics"]])))
                                for k, v in val_res_ranking.items():
                                    self.writer.add_scalar(k, val_res_ranking[k], i)
                                if self.val_decision != "loss":
                                    val_dec = val_res_ranking["val_FULL_{}".format(self.val_decision)]
                                    print("~~VAL_FULL_{}_iter{}: {}".format(self.val_decision, i, val_dec, 4))
                            self.validation_res_handler.write("\n")
                            if "early_stopping" in self.config:
                                self.saver(val_dec, self, i)
                                if self.saver.stop:  # meaning we reach the early stopping criterion
                                    print("== EARLY STOPPING AT ITER {}".format(i))
                                    with open_dict(self.config):
                                        self.config["stop_iter"] = i
                                    break
                            else:
                                self.saver(val_dec, self, i)
                        # same for test (if test loader !):
                        if self.test_loader is not None:
                            # no use for now
                            pass
                if not self.validation:
                    self.save_checkpoint(step=i, perf=loss, is_best=True)
        if not self.validation:
            # when no validation, finally save the final model (last epoch)
            self.save_checkpoint(step=i, perf=loss, is_best=True)
        self.save_checkpoint(step=i, perf=loss, is_best=False, final_checkpoint=True)  # save the last anyway


class SiameseTransformerTrainer(TransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        # for this trainer, the batch contains query, pos doc and neg doc HF formatted inputs
        q_kwargs = parse(batch, "q")
        d_pos_kwargs = parse(batch, "pos")
        d_neg_kwargs = parse(batch, "neg")
        d_pos_args = {"q_kwargs": q_kwargs, "d_kwargs": d_pos_kwargs}
        d_neg_args = {"q_kwargs": q_kwargs, "d_kwargs": d_neg_kwargs}
        if "augment_pairs" in self.config:
            if self.config["augment_pairs"] == "in_batch_negatives":
                d_pos_args["score_batch"] = True  # meaning that for the POSITIVE documents in the batch, we will
                # actually compute all the scores w.r.t. the queries in the batch
            else:
                raise NotImplementedError
        with torch.cuda.amp.autocast() if self.fp16 else amp.NullContextManager():
            out_pos = self.model(**d_pos_args)
            out_neg = self.model(**d_neg_args)
        out = {}
        for k, v in out_pos.items():
            out["pos_{}".format(k)] = v
        for k, v in out_neg.items():
            out["neg_{}".format(k)] = v
        if "teacher_p_score" in batch:  # distillation pairs dataloader
            out["teacher_pos_score"] = batch["teacher_p_score"]
            out["teacher_neg_score"] = batch["teacher_n_score"]
        return out

    def evaluate_loss(self, data_loader):
        """loss evaluation
        """
        out_d = defaultdict(float)
        for batch in data_loader:
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            out = self.forward(batch)
            val_ranking_loss = self.loss(out).mean().item()
            out_d["val_ranking_loss"] += val_ranking_loss
            if self.regularizer is not None:
                if "train" in self.regularizer:
                    total_loss = val_ranking_loss
                    for reg in self.regularizer["train"]:
                        lambda_q = self.regularizer["train"][reg]["lambdas"]["lambda_q"].get_lambda() if "lambda_q" in \
                                                                                                         self.regularizer[
                                                                                                             "train"][
                                                                                                             reg][
                                                                                                             "lambdas"] else False
                        lambda_d = self.regularizer["train"][reg]["lambdas"]["lambda_d"].get_lambda() if "lambda_d" in \
                                                                                                         self.regularizer[
                                                                                                             "train"][
                                                                                                             reg][
                                                                                                             "lambdas"] else False
                        targeted_rep = self.regularizer["train"][reg]["targeted_rep"]
                        r_loss = 0
                        if lambda_q:
                            r_loss += (self.regularizer["train"][reg]["loss"](
                                out["pos_q_{}".format(targeted_rep)]) * lambda_q).mean().item()

                        if lambda_d:
                            r_loss += ((self.regularizer["train"][reg]["loss"](
                                out["pos_d_{}".format(targeted_rep)]) * lambda_d).mean().item() + (
                                               self.regularizer["train"][reg]["loss"](
                                                   out["neg_d_{}".format(targeted_rep)]) * lambda_d).mean().item()) / 2
                        out_d["val_{}_loss".format(reg)] += r_loss
                        total_loss += r_loss
                    out_d["val_total_loss"] += total_loss
        return {key: value / len(data_loader) for key, value in out_d.items()}

    def evaluate_full_ranking(self, i):
        """full ranking evaluation
        """
        out_d = {}
        eval_full = self.validation_evaluator.index_and_retrieve(i)
        run = eval_full["retrieval"]
        for metric in self.validation_metrics:
            out_d["val_FULL_{}".format(metric)] = init_eval(metric)(dict(run), self.full_rank_qrel)
        if "stats" in eval_full:
            for k, v in eval_full["stats"].items():
                out_d["val_FULL_{}".format(k)] = v
        return out_d

    def save_checkpoint(self, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        # it is practical (although redundant) to save model weights using huggingface API, because if the model has
        # no other params, we can reload it easily with .from_pretrained()
        output_dir = os.path.join(self.config["checkpoint_dir"], "model")
        model_to_save.transformer_rep.transformer.save_pretrained(output_dir)
        tokenizer = model_to_save.transformer_rep.tokenizer
        tokenizer.save_pretrained(output_dir)
        if model_to_save.transformer_rep_q is not None:
            output_dir_q = os.path.join(self.config["checkpoint_dir"], "model_q")
            model_to_save.transformer_rep_q.transformer.save_pretrained(output_dir_q)
            tokenizer = model_to_save.transformer_rep_q.tokenizer
            tokenizer.save_pretrained(output_dir_q)
        super().save_checkpoint(**kwargs)
        

class SiameseTransformerAdapterTrainer(SiameseTransformerTrainer):
    def __init__(self, adapter_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        # Set the defaults for loading/ saving model & adapters
        model = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        if isinstance(model.transformer_rep.transformer, PreTrainedModel):
            model.transformer_rep.transformer.train_adapter(model.transformer_rep.transformer.active_adapters)
            if model.transformer_rep.transformer.active_adapters:
                # Check if training AdapterFusion
                self.train_adapter_fusion = (
                    isinstance(model.transformer_rep.transformer.active_adapters, Fuse)
                    or isinstance(model.transformer_rep.transformer.active_adapters, AdapterCompositionBlock)
                    and any([isinstance(child, Fuse) for child in model.transformer_rep.transformer.active_adapters.children])
                )
        
        if isinstance(model.transformer_rep_q, PreTrainedModel):
            model.transformer_rep_q.transformer.train_adapter(model.transformer_rep_q.transformer.active_adapters)
       
        if model.transformer_rep.transformer.active_adapters is None and model.transformer_rep_q.transformer.active_adapters is None:
            raise ValueError(
                "Expected a model with an active adapter setup."
                "If you want to fully finetune the model use the SiameseTransformerTrainer class."
            )
            
    def _load_adapters(self, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "adapter_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_adapter(os.path.join(os.path.join(resume_from_checkpoint, file_name)))
                    adapter_loaded = True
        return adapter_loaded

    def _load_adapter_fusions(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," in file_name:
                    self.model.load_adapter_fusion(os.path.join(resume_from_checkpoint, file_name))
           
    def save_checkpoint(self, step, perf, is_best=True, final_checkpoint=False, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        with open_dict(self.config):
            self.config["ckpt_step"] = step
        state = {"step": step,
                 "perf": perf,
                 "model_state_dict": model_to_save.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "config": self.config,
                 "regularizer": self.regularizer,
                 }
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
            state["scheduler_state_dict"] = scheduler_state_dict
        
        # it is practical (although redundant) to save model weights using huggingface API, because if the model has
        # no other params, we can reload it easily with .from_pretrained()
        output_dir = os.path.join(self.config["checkpoint_dir"], "model")
        #model_to_save.transformer_rep.transformer.save_pretrained(output_dir)
        
        #Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not final_checkpoint:
            if not isinstance(model_to_save.transformer_rep.transformer, PreTrainedModel):
                if isinstance(unwrap_model(model_to_save.transformer_rep.transformer), PreTrainedModel):
                    if state is None:
                        state_dict = model_to_save.transformer_rep.transformer.state_dict()
                    unwrap_model(model_to_save.transformer_rep.transformer).save_pretrained(output_dir, state_dict=state_dict)
                else:
                    #logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if state is None:
                        state_dict = model_to_save.transformer_rep.transformer.state_dict()
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                # rename last:
                if os.path.exists(os.path.join(self.checkpoint_dir, "model_ckpt/model_last/model_last.tar")):
                    last_config = torch.load(os.path.join(self.checkpoint_dir, "model_ckpt/model_last/model_last.tar"))
                    step_last_config = last_config["step"]
                    if not os.path.exists(os.path.join(self.checkpoint_dir, f"model_ckpt/model_ckpt_{step_last_config}")):
                        os.makedirs(os.path.join(self.checkpoint_dir, f"model_ckpt/model_ckpt_{step_last_config}"))
                    os.rename(os.path.join(self.checkpoint_dir, "model_ckpt/model_last/model_last.tar"),
                            os.path.join(self.checkpoint_dir, f"model_ckpt/model_ckpt_{step_last_config}/model_ckpt_{step_last_config}.tar"))
                if not os.path.exists(os.path.join(self.checkpoint_dir, "model_ckpt/model_last")):
                    os.makedirs(os.path.join(self.checkpoint_dir, "model_ckpt/model_last"))
                torch.save(state, os.path.join(self.checkpoint_dir,  "model_ckpt/model_last/model_last.tar"))
                model_to_save.transformer_rep.transformer.save_all_adapters(os.path.join(self.checkpoint_dir, "model_ckpt/model_last"))
                if model_to_save.transformer_rep_q is not None:
                    model_to_save.transformer_rep_q.transformer.save_all_adapters(os.path.join(self.checkpoint_dir, "model_ckpt/model_last"))
                if hasattr(model_to_save.transformer_rep.transformer, "heads"):
                    model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model_ckpt/model_last"))
                    if model_to_save.transformer_rep_q is not None:
                            model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model_ckpt/model_last"))
                                           
                if is_best:
                    if not os.path.exists(os.path.join(self.checkpoint_dir, "model")):
                        os.makedirs(os.path.join(self.checkpoint_dir, "model"))
                    torch.save(state, os.path.join(self.checkpoint_dir, "model/model.tar"))
                    model_to_save.transformer_rep.transformer.save_all_adapters(os.path.join(self.checkpoint_dir, "model"))
                    if self.train_adapter_fusion:
                        model_to_save.transformer_rep.transformer.save_all_adapter_fusions(os.path.join(self.checkpoint_dir, "model"))
                        if model_to_save.transformer_rep_q is not None:
                            model_to_save.transformer_rep.transformer.save_all_adapter_fusions(os.path.join(self.checkpoint_dir, "model"))
                    if hasattr(model_to_save.transformer_rep.transformer, "heads"):
                        model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model"))
                        if model_to_save.transformer_rep_q is not None:
                                model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model"))
                    
                # remove oldest checkpoint (by default only keep the last 3):
                remove_old_ckpt_adapters(os.path.join(self.checkpoint_dir, "model_ckpt"), k=3)
        else:
            if not os.path.exists(os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint/")):
                        os.makedirs(os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint/"))
            torch.save(state, os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint/model_final_checkpoint.tar"))
            model_to_save.transformer_rep.transformer.save_all_adapters(os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint"))
            if model_to_save.transformer_rep_q is not None:
                model_to_save.transformer_rep_q.transformer.save_all_adapters(os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint"))
            if hasattr(model_to_save.transformer_rep.transformer, "heads"):
                model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint"))
                if model_to_save.transformer_rep_q is not None:
                        model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model_ckpt/model_final_checkpoint"))
            if self.overwrite_final:
                torch.save(state, os.path.join(self.checkpoint_dir, "model/model.tar"))
                if model_to_save.transformer_rep_q is not None:
                    model_to_save.transformer_rep_q.transformer.save_all_adapters(os.path.join(self.checkpoint_dir, "model"))
                if hasattr(model_to_save.transformer_rep.transformer, "heads"):
                    model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model"))
                    if model_to_save.transformer_rep_q is not None:
                            model_to_save.transformer_rep.transformer.save_all_heads(os.path.join(self.checkpoint_dir, "model"))
        tokenizer = model_to_save.transformer_rep.tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        #torch.save(self.args, os.path.join(output_dir, "training_args.bin"))                
        #super().save_checkpoint(**kwargs)
