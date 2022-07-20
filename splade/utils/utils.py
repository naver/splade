import os
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..losses.pairwise import DistilKLLoss, PairwiseNLL, DistilMarginMSE, InBatchPairwiseNLL
from ..losses.pointwise import BCEWithLogitsLoss


def parse(d, name):
    return {k.replace(name + "_", ""): v for k, v in d.items() if name in k}


def rename_keys(d, prefix):
    return {prefix + "_" + k: v for k, v in d.items()}


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def set_seed(seed):
    """see: https://twitter.com/chaitjo/status/1394936019506532353/photo/1
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def restore_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    # strict = False => it means that we just load the parameters of layers which are present in both and
    # ignores the rest
    if len(missing_keys) > 0:
        print("~~ [WARNING] MISSING KEYS WHILE RESTORING THE MODEL ~~")
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print("~~ [WARNING] UNEXPECTED KEYS WHILE RESTORING THE MODEL ~~")
        print(unexpected_keys)
    print("restoring model:", model.__class__.__name__)


def remove_old_ckpt(dir_, k):
    ckpt_names = os.listdir(dir_)
    if len(ckpt_names) <= k:
        pass
    else:
        ckpt_names.remove("model_last.tar")
        steps = []
        for ckpt_name in ckpt_names:
            steps.append(int(ckpt_name.split(".")[0].split("_")[-1]))
        oldest = sorted(steps)[0]
        print("REMOVE", os.path.join(dir_, "model_ckpt_{}.tar".format(oldest)))
        os.remove(os.path.join(dir_, "model_ckpt_{}.tar".format(oldest)))


def generate_bow(input_ids, output_dim, device, values=None):
    """from a batch of input ids, generates batch of bow rep
    """
    bs = input_ids.shape[0]
    bow = torch.zeros(bs, output_dim).to(device)
    if values is None:
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = 1
    else:
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = values
    return bow


def normalize(tensor, eps=1e-9):
    """normalize input tensor on last dimension
    """
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)


def get_dataset_name(path):
    # small (hard-coded !) snippet to get a dataset name from a Q_COLLECTION_PATH or a EVAL_QREL_PATH (full paths)
    if "TREC_DL_2019" in path:
        return "TREC_DL_2019"
    elif "trec2020" in path or "TREC_DL_2020" in path:
        return "TREC_DL_2020"
    elif "msmarco" in path:
        if "train_queries" in path:
            return "MSMARCO_TRAIN"
        else:
            return "MSMARCO"
    elif "MSMarco-v2" in path:
        if "dev_1" in path:
            return "MSMARCO_v2_dev1"
        else:
            assert "dev_2" in path
            return "MSMARCO_v2_dev2"
    elif "toy" in path:
        return "TOY"
    else:
        return "other_dataset"


def get_initialize_config(exp_dict: DictConfig, train=False):
    # delay import to reduce dependencies
    from ..utils.hydra import hydra_chdir
    hydra_chdir(exp_dict)
    exp_dict["init_dict"]["fp16"] = exp_dict["config"].get("fp16", False)
    config = exp_dict["config"]
    init_dict = exp_dict["init_dict"]
    if train:
        os.makedirs(exp_dict.config.checkpoint_dir, exist_ok=True)
        OmegaConf.save(config=exp_dict, f=os.path.join(exp_dict.config.checkpoint_dir, "config.yaml"))
        model_training_config = None
    else:
        if config.pretrained_no_yamlconfig:
            model_training_config = config
        else:
            model_training_config = OmegaConf.load(os.path.join(config["checkpoint_dir"], "config.yaml"))["config"]
    return exp_dict, config, init_dict, model_training_config


def get_loss(config):
    if config["loss"] == "PairwiseNLL":
        loss = PairwiseNLL()
    elif config["loss"] == "DistilMarginMSE":
        loss = DistilMarginMSE()
    elif config["loss"] == "KlDiv":
        loss = DistilKLLoss()
    elif config["loss"] == "InBatchPairwiseNLL":
        loss = InBatchPairwiseNLL()
    elif config["loss"] == "BCE":
        loss = BCEWithLogitsLoss()
    else:
        raise NotImplementedError("provide valid loss")
    return loss


def set_seed_from_config(config):
    if "random_seed" in config:
        random_seed = config["random_seed"]
    else:
        random_seed = 123
    set_seed(random_seed)
    return random_seed
