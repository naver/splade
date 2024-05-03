import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import AnseriniCollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import EncodeAnserini
from .utils.utils import get_initialize_config
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from .datasets.datasets import BeirDatasetAnserini
import os

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def index(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    model = get_model(config, init_dict)

    if model_training_config["matching_type"] == "splade":
        quantization_factor_doc = exp_dict["quantization_factor_document"]
        quantization_factor_query = exp_dict["quantization_factor_query"]
    elif model_training_config["matching_type"] == "splade_doc":
        quantization_factor_doc = exp_dict["quantization_factor_document"]
        quantization_factor_query = 1
    else:
        raise NotImplementedError

    # Download and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        exp_dict["beir"]["dataset"])
    out_dir = exp_dict["beir"]["dataset_path"]
    data_path = util.download_and_unzip(url, out_dir)

    config["index_dir"] = os.path.join(config["index_dir"], "beir", exp_dict["beir"]["dataset"])
    os.makedirs(config["index_dir"], exist_ok=True)

    out_dir_2 = os.path.join(config["out_dir"], "beir", exp_dict["beir"]["dataset"])
    config["out_dir"] = os.path.join(config["out_dir"], "beir", exp_dict["beir"]["dataset"],"docs")
    os.makedirs(config["out_dir"], exist_ok=True)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=exp_dict["beir"].get("split","test"))

    d_collection = BeirDatasetAnserini(corpus, information_type="document")
    q_collection = BeirDatasetAnserini(queries, information_type="query")


    d_loader = AnseriniCollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"],
                                        batch_size=config["index_retrieve_batch_size"],
                                        shuffle=False, num_workers=4)
    evaluator = EncodeAnserini(model, config)
    evaluator.index(d_loader, quantization_factor=quantization_factor_doc)
    config["out_dir"] = out_dir_2
    q_loader = AnseriniCollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"],
                                        batch_size=config["index_retrieve_batch_size"],
                                        shuffle=False, num_workers=4)
    evaluator = EncodeAnserini(model, config, input_type="query")
    evaluator.index(q_loader, quantization_factor=quantization_factor_query)


if __name__ == "__main__":
    index()
