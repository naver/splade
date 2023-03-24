from dataclasses import dataclass, field
from typing import Optional

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

    adapter_name: str = field(
        metadata={"help": "Adapter name"},
        default=None
    )

    adapter_config: str = field(
        metadata={"help": "Adapter Config/Type : {houlsby, pfeiffer, prefixtuning, LoRA"},
        default=None
    )

    load_adapter: str = field(
        metadata={"help": "Adapter to load"},
        default=None
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    distillation: bool = field(
        metadata={"help": "Use distillation data"},
        default=False
    )
    result_path: Optional[str] = field(
        metadata={"help": "Path to result file"},
        default=None
#        default="/scratch/1/user/slupart/learn2index/exp/ALL/splade_distil_L1_splade_cocon/out_train/other_dataset/run.json"
    )

    scores: Optional[str] = field(
        metadata={"help": "Path to score file"},
        default=None
#        default="/nfs/data/neuralsearch/msmarco/scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
    )

    document_dir: str = field(
        metadata={"help": "Path to doc file"},
        default="/nfs/data/neuralsearch/msmarco/documents/raw.tsv"
    )
    query_dir: str = field(
        metadata={"help": "Path to query file"},
        default="/nfs/data/neuralsearch/msmarco/training_queries/raw.tsv"
    )
    qrels_path: str = field(
        metadata={"help": "Path to qrels"},
        default="/nfs/data/neuralsearch/msmarco/training_queries/qrels.json"
    )

    negatives_path: str = field(
        metadata={"help": "Path to negatives"},
        default=None,
#        default="/scratch/1/user/classanc/test_reranker/final_negatives.pkl.gz",
    )


    n_negatives: int = field(
        metadata={"help": "Negatives per query"},
        default=7,
    )

    margin: int = field(
        metadata={"help": "Margin to remove possible false negatives"},
        default=3,
    )
    n_queries: int = field(
        metadata={"help": "number of queries"},
        default=-1,
    )


    top_k: int = field(
        metadata={"help": "Top to filter"},
        default=-1,
    )



@dataclass
class LocalTrainingArguments(TrainingArguments):
    """
    Arguments for training.
    """

    output_dir: str = field(
        metadata={"help": "Output path dir"},
        default="models/t5-small/output"
    )

    per_device_train_batch_size: int = field(
        metadata={"help": "BS per device"},
        default=2
    )
    
    logging_steps: int = field(
        metadata={"help": "Steps to log"},
        default=10
    )

    mse_margin: bool = field(
        metadata={"help": "Use mse margin for distillation"},
        default=False
    )

    l0d: float = field(
        metadata={"help": "Use mse margin for distillation"},
        default=5e-4
    )

    l0q: float = field(
        metadata={"help": "Use mse margin for distillation"},
        default=5e-4
    )

    save_total_limit: int = field(
        metadata={"help": "Total number of checkpoints to save. Deletes older checkpoints"},
        default=-1
    )
