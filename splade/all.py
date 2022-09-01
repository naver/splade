import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.flops import flops
from splade.index import index
from splade.retrieve import retrieve_evaluate
from splade.train import train
from splade.utils.hydra import hydra_chdir
from splade.utils.index_figure import index_figure


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def train_index_retrieve(exp_dict: DictConfig):
    hydra_chdir(exp_dict)

    train(exp_dict)
    index(exp_dict)
    retrieve_evaluate(exp_dict)
    flops(exp_dict)
    index_figure(exp_dict)


if __name__ == "__main__":
    train_index_retrieve()
