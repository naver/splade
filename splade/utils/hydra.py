import os

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


def hydra_chdir(exp_dict):
    print(OmegaConf.to_yaml(exp_dict))
    try:
        os.chdir(get_original_cwd())
    except ValueError:
        # hydra manual init, nothing to do
        pass
