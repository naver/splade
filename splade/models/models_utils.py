from omegaconf import DictConfig

from ..models.transformer_rep import Splade, SpladeDoc


def get_model(config: DictConfig, init_dict: DictConfig):
    # no need to reload model here, it will be done later
    # (either in train.py or in Evaluator.__init__()

    model_map = {
        "splade": Splade,
        "splade_doc": SpladeDoc
    }
    try:
        model_class = model_map[config["matching_type"]]
    except KeyError:
        raise NotImplementedError("provide valid matching type ({})".format(config["matching_type"]))
    return model_class(**init_dict)
