from dataclasses import dataclass, field
from typing import Optional, Literal

from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    max_length: int = field(
        metadata={"help": "Max length for sequences"},
        default=128
    )

    shared_weights: bool = field(
        metadata={"help": "Share weights between doc and query encoder"},
        default=True
    )
    splade_doc: bool = field(
        metadata={"help": "Is spladedoc"},
        default=False
    )
    
    model_q: str = field(
        metadata={"help": "Path to pretrained model or model identifier for the query from huggingface.co/models"},
        default=None
    )
    
    dense_pooling: bool = field(
        metadata={"help": "dense pooling"},
        default="cls"
    )

    dense: bool = field(
        metadata={"help": "Dense model"},
        default=False
    )

    # adapter_name: str = field(
    #     metadata={"help": "Adapter name"},
    #     default=None
    # )

    # adapter_config: str = field(
    #     metadata={"help": "Adapter Config/Type : {houlsby, pfeiffer, prefixtuning, LoRA"},
    #     default=None
    # )

    # load_adapter: str = field(
    #     metadata={"help": "Adapter to load"},
    #     default=None
    # )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    training_data_type:Optional[str] = field(
        metadata={"help": "data format (json, pkl_dict, saved_pkl, trec, triplets)"},
        default=None 
    )
    training_data_path: Optional[str] = field(
        metadata={"help": "Path to training file"},
        default=None
    )

    document_dir: str = field(
        metadata={"help": "Path to collection file"},
        default=None
    )
    query_dir: str = field(
        metadata={"help": "Path to query file"},
        default=None
    )
    qrels_path: str = field(
        metadata={"help": "Path to qrels"},
        default=None
    )

    n_negatives: int = field(
        metadata={"help": "Negatives per query"},
        default=4,
    )


    n_queries: int = field(
        metadata={"help": "number of queries"},
        default=-1,
    )




@dataclass
class LocalTrainingArguments(TrainingArguments):
    """
    SPLADE Arguments for training.
    """
    output_dir: str = field(
        # Rv: output of what ? reporting?
        metadata={"help": "Output path dir"},
        default=None
    )

    training_loss: str = field(
        metadata={"help": "Which losses to use: contrastive, kldiv, mse_margin, kldiv_mse_margin_with_weights, kldiv_mse_margin_without_weights, kldiv_contrastive_without_weights, kldiv_contrastive_with_weights"},
        default="kldiv_contrastive_with_weights"
    )

    l0d: float = field(
        metadata={"help": "lambda for document"},
        default=5e-4
    )

    l0q: float = field(
        metadata={"help": "lambda for query"},
        default=5e-4
    )

    T_d: int = field(
        metadata={"help": "Exponential FLOPS growth for lambda_d"},
        default=0
    )

    T_q: int = field(
        metadata={"help": "Exponential FLOPS growth for lambda_q"},
        default=0
    )

    top_d: int = field(
        metadata={"help": "TOP_k document pruning"},
        default=-1
    )

    top_q: int = field(
        metadata={"help": "TOP_k query pruning"},
        default=-1
    )

    lexical_type: str = field(
        metadata={"help": "Type of splade lexical to do: none, document, query or both"},
        default="none",
    )


