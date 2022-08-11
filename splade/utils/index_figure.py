import json
import os

import numpy as np
from omegaconf import DictConfig

import hydra
from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def index_figure(exp_dict: DictConfig):
    import matplotlib.pyplot as plt
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)
    index_folder = config.index_dir
    index_file = os.path.join(index_folder, "index_dist.json")
    figure_file = os.path.join(index_folder, "index_dist.png")
    index_dist = json.load(open(index_file))

    sorted_dist = -np.array(sorted(-np.array(list(index_dist.values()))))

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(sorted_dist)
    ax.set_yscale("log")
    ax.set_title("Index distribution (size of posting list)")
    ax.set_xlabel("Token in decreasing number of documents")
    ax.set_ylabel("Number of documents")
    plt.savefig(figure_file)
    print(figure_file)


if __name__ == "__main__":
    index_figure()
