import os

import hydra
import torch
from omegaconf import DictConfig, open_dict
from torch.utils import data

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader, SiamesePairsDataLoader, DistilSiamesePairsDataLoader
from .datasets.datasets import PairsDatasetPreLoad, DistilPairsDatasetPreLoad, MsMarcoHardNegatives, \
    CollectionDatasetPreLoad
from .losses.regularization import init_regularizer, RegWeightScheduler
from .models.models_utils import get_model
from .optim.bert_optim import init_simple_bert_optim
from .tasks.transformer_evaluator import SparseApproxEvalWrapper
from .tasks.transformer_trainer import SiameseTransformerTrainer
from .utils.utils import set_seed, restore_model, get_initialize_config, get_loss, set_seed_from_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def train(exp_dict: DictConfig):
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
    model = get_model(config, init_dict)
    random_seed = set_seed_from_config(config)

    optimizer, scheduler = init_simple_bert_optim(model, lr=config["lr"], warmup_steps=config["warmup_steps"],
                                                  weight_decay=config["weight_decay"],
                                                  num_training_steps=config["nb_iterations"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ################################################################
    # CHECK IF RESUME TRAINING
    ################################################################
    iterations = (1, config["nb_iterations"] + 1)  # tuple with START and END
    regularizer = None
    if os.path.exists(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar")):
        print("@@@@ RESUMING TRAINING @@@")
        print("WARNING: change seed to change data order when restoring !")
        set_seed(random_seed + 666)
        if device == torch.device("cuda"):
            ckpt = torch.load(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar"))
        else:
            ckpt = torch.load(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar"), map_location=device)
        print("starting from step", ckpt["step"])
        print("{} remaining iterations".format(iterations[1] - ckpt["step"]))
        iterations = (ckpt["step"] + 1, config["nb_iterations"])
        restore_model(model, ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if device == torch.device("cuda"):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "regularizer" in ckpt:
            print("loading regularizer")
            regularizer = ckpt.get("regularizer", None)

    if torch.cuda.device_count() > 1:
        print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    loss = get_loss(config)

    # initialize regularizer dict
    if "regularizer" in config and regularizer is None:  # else regularizer is loaded
        output_dim = model.module.output_dim if hasattr(model, "module") else model.output_dim
        regularizer = {"eval": {"L0": {"loss": init_regularizer("L0")},
                                "sparsity_ratio": {"loss": init_regularizer("sparsity_ratio",
                                                                            output_dim=output_dim)}},
                       "train": {}}
        if config["regularizer"] == "eval_only":
            # just in the case we train a model without reg but still want the eval metrics like L0
            pass
        else:
            for reg in config["regularizer"]:
                temp = {"loss": init_regularizer(config["regularizer"][reg]["reg"]),
                        "targeted_rep": config["regularizer"][reg]["targeted_rep"]}
                d_ = {}
                if "lambda_q" in config["regularizer"][reg]:
                    d_["lambda_q"] = RegWeightScheduler(config["regularizer"][reg]["lambda_q"],
                                                        config["regularizer"][reg]["T"])
                if "lambda_d" in config["regularizer"][reg]:
                    d_["lambda_d"] = RegWeightScheduler(config["regularizer"][reg]["lambda_d"],
                                                        config["regularizer"][reg]["T"])
                temp["lambdas"] = d_  # it is possible to have reg only on q or d if e.g. you only specify lambda_q
                # in the reg config
                # targeted_rep is just used to indicate which rep to constrain (if e.g. the model outputs several
                # representations)
                # the common case: model outputs "rep" (in forward) and this should be the value for this targeted_rep
                regularizer["train"][reg] = temp

    # fix for current in batch neg losses that break on last batch
    if config["loss"] in ("InBatchNegHingeLoss", "InBatchPairwiseNLL"):
        drop_last = True
    else:
        drop_last = False

    if exp_dict["data"].get("type", "") == "triplets":
        data_train = PairsDatasetPreLoad(data_dir=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets"
    elif exp_dict["data"].get("type", "") == "triplets_with_distil":
        data_train = DistilPairsDatasetPreLoad(data_dir=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets_with_distil"
    elif exp_dict["data"].get("type", "") == "hard_negatives":
        data_train = MsMarcoHardNegatives(
            dataset_path=exp_dict["data"]["TRAIN"]["DATASET_PATH"],
            document_dir=exp_dict["data"]["TRAIN"]["D_COLLECTION_PATH"],
            query_dir=exp_dict["data"]["TRAIN"]["Q_COLLECTION_PATH"],
            qrels_path=exp_dict["data"]["TRAIN"]["QREL_PATH"])
        train_mode = "triplets_with_distil"
    else:
        raise ValueError("provide valid data type for training")

    val_loss_loader = None  # default
    if "VALIDATION_SIZE_FOR_LOSS" in exp_dict["data"]:
        print("initialize loader for validation loss")
        print("split train, originally {} pairs".format(len(data_train)))
        data_train, data_val = torch.utils.data.random_split(data_train, lengths=[
            len(data_train) - exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"],
            exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"]])
        print("train: {} pairs ~~ val: {} pairs".format(len(data_train), len(data_val)))
        if train_mode == "triplets":
            val_loss_loader = SiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
                                                     shuffle=False,
                                                     num_workers=4,
                                                     tokenizer_type=config["tokenizer_type"],
                                                     max_length=config["max_length"], drop_last=drop_last)
        elif train_mode == "triplets_with_distil":
            val_loss_loader = DistilSiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
                                                           shuffle=False,
                                                           num_workers=4,
                                                           tokenizer_type=config["tokenizer_type"],
                                                           max_length=config["max_length"], drop_last=drop_last)
        else:
            raise NotImplementedError

    if train_mode == "triplets":
        train_loader = SiamesePairsDataLoader(dataset=data_train, batch_size=config["train_batch_size"], shuffle=True,
                                              num_workers=4,
                                              tokenizer_type=config["tokenizer_type"],
                                              max_length=config["max_length"], drop_last=drop_last)
    elif train_mode == "triplets_with_distil":
        train_loader = DistilSiamesePairsDataLoader(dataset=data_train, batch_size=config["train_batch_size"],
                                                    shuffle=True,
                                                    num_workers=4,
                                                    tokenizer_type=config["tokenizer_type"],
                                                    max_length=config["max_length"], drop_last=drop_last)
    else:
        raise NotImplementedError

    val_evaluator = None
    if "VALIDATION_FULL_RANKING" in exp_dict["data"]:
        with open_dict(config):
            config["val_full_rank_qrel_path"] = exp_dict["data"]["VALIDATION_FULL_RANKING"]["QREL_PATH"]
        full_ranking_d_collection = CollectionDatasetPreLoad(
            data_dir=exp_dict["data"]["VALIDATION_FULL_RANKING"]["D_COLLECTION_PATH"], id_style="row_id")
        full_ranking_d_loader = CollectionDataLoader(dataset=full_ranking_d_collection,
                                                     tokenizer_type=config["tokenizer_type"],
                                                     max_length=config["max_length"],
                                                     batch_size=config["eval_batch_size"],
                                                     shuffle=False, num_workers=4)
        full_ranking_q_collection = CollectionDatasetPreLoad(
            data_dir=exp_dict["data"]["VALIDATION_FULL_RANKING"]["Q_COLLECTION_PATH"], id_style="row_id")
        full_ranking_q_loader = CollectionDataLoader(dataset=full_ranking_q_collection,
                                                     tokenizer_type=config["tokenizer_type"],
                                                     max_length=config["max_length"], batch_size=1,
                                                     # TODO fix: bs currently set to 1
                                                     shuffle=False, num_workers=4)
        val_evaluator = SparseApproxEvalWrapper(model,
                                                config={"top_k": exp_dict["data"]["VALIDATION_FULL_RANKING"]["TOP_K"],
                                                        "out_dir": os.path.join(config["checkpoint_dir"],
                                                                                "val_full_ranking")
                                                        },
                                                collection_loader=full_ranking_d_loader,
                                                q_loader=full_ranking_q_loader,
                                                restore=False)

    # #################################################################
    # # TRAIN
    # #################################################################
    print("+++++ BEGIN TRAINING +++++")
    trainer = SiameseTransformerTrainer(model=model, iterations=iterations, loss=loss, optimizer=optimizer,
                                        config=config, scheduler=scheduler,
                                        train_loader=train_loader, validation_loss_loader=val_loss_loader,
                                        validation_evaluator=val_evaluator,
                                        regularizer=regularizer)
    trainer.train()


if __name__ == "__main__":
    train()
