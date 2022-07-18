import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import TextCollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import EncodeAnserini
from .utils.utils import get_initialize_config


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
    d_collection = CollectionDatasetPreLoad(data_dir=exp_dict["data"]["COLLECTION_PATH"], id_style="row_id")
    d_loader = TextCollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"],
                                        batch_size=config["index_retrieve_batch_size"],
                                        shuffle=False, num_workers=4)
    evaluator = EncodeAnserini(model, config)
    evaluator.index(d_loader, quantization_factor=quantization_factor_doc)
    q_collection = CollectionDatasetPreLoad(data_dir=exp_dict["data"]["Q_COLLECTION_PATH"][0], id_style="row_id")
    q_loader = TextCollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"],
                                        batch_size=config["index_retrieve_batch_size"],
                                        shuffle=False, num_workers=4)
    evaluator = EncodeAnserini(model, config, input_type="query")
    evaluator.index(q_loader, quantization_factor=quantization_factor_query)


if __name__ == "__main__":
    index()
