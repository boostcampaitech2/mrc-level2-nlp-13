from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# monologg/koelectra-base-v3-finetuned-korquad 
# epoch 10 - 60.8
# epoch 20 - 
# monologg/kobigbird-bert-base 
# epoch 10 - 61.6
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="monologg/koelectra-base-v3-finetuned-korquad",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default="monologg/koelectra-base-v3-finetuned-korquad",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="monologg/koelectra-base-v3-finetuned-korquad",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    kind_of_retrieval: str = field(
        default='Sparse', #SparseDense
        metadata={"help": "Kind of retrieval."},
    )
    num_clusters: int = field(
        default=128, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=35,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    use_validation_data: bool = field(
        default=False,
        metadata={"help": "Whether to train with validation set"},
    )

@dataclass
class CustomArguments:
    """
    Arguments wandb and custom settings for training
    """
    # wandb
    use_wandb: bool = field(
        default=True, metadata={"help": "Whether to use Wandb"}
    )
    entity_name: str = field(
        default="clue",
        metadata={
            "help": "Your entity name in WandB E.g. clue or KyunghyunLim, ..."
        },
    )
    project_name: str = field(
        default="mrc_test",
        metadata={
            "help": "Your project name in WandB E.g. LKH, Readers, ..."
        },
    )
    wandb_run_name: str = field(
        default="elastic-kobigbird",
        metadata={
            "help": "run name in WandB E.g. Bart_v0.1, Roberta_v0.1, DPR_Bert_v0.1"
        },
    )
    description: str = field(
        default="Roberta-large epoch 10",
        metadata={
            "help": "Explain your specific experiments settings"
        },
    )

    # Training
    epochs: int = field(
        default = 10,
        metadata={
            "help": "Training epoch"
        },
    )
    custom_learning_rate: float = field(
        default = 5e-5,
        metadata={
            "help": "Training learning rate"
        },
    )
    train_batch_size: int = field(
        default = 16,
        metadata={
            "help": "Training batch size"
        },
    )
    valid_batch_size: int = field(
        default = 16,
        metadata={
            "help": "Validation batch size"
        },
    )
    accumulation_step: int = field(
        default = 10,
        metadata={
            "help": "Training accumulation step"
        },
    )
    sample_logging_step: int = field(
        default = 50,
        metadata={
            "help": "Print samples for each set value."
        },
    )
    sample_logging_amount: int = field(
        default = 5,
        metadata={
            "help": "Number of samples you want to print"
        },
    )
    overwite: bool = field(
        default = True, # False
        metadata={
            "help": "whether overwrite or not"
        },
    )

@dataclass
class QuestionGenerationArguments:
    """
    Arguments for Question Generation
    """
    data_path: Optional[str] = field(
        default= "../data/",
        metadata={"help": "path for data"},
    )

    context_path: Optional[str] = field(
        default= "wikipedia_documents.json",
        metadata={"help": "exact path for wiki json file"},
    )