##################
# Import modules #
##################

from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Any, Dict, List, Optional
import json
import wandb

########################
# Set global variables #
########################

MODEL_DIR = "tuned_models/"
INFERENCE_DIR = "predictions/"
CONFIG_DIR = "configs/"
LOG_DIR = "logs/"

#######################
# Classes & Functions #
#######################

def get_info(config_json_path : str):
    with open(config_json_path, "r") as file:
        info = json.load(file)
    return info

def check_bool(value):
    if value in ["True","False"]:
        if value == "True":
            return True
        elif value == "False":
            return False
    else:
        return value

def train_config_setting(
    config
):
    '''
        Reader Train config 변수 추가
    '''

    info = get_info(config.config_file_path)
    model_args = DataArguments()
    model_args, data_args, training_args, custom_args = \
        ModelArguments(), DataArguments(), TrainingArguments(config.model_name), CustomArguments()
    
    # 모델 관련 정보 저장
    for val_name, value in info['model_args'].items():
        value = check_bool(value)
        setattr(model_args, val_name, value)

    # 데이터 관련 정보 저장
    for val_name, value in info['data_args'].items():
        value = check_bool(value)
        setattr(data_args, val_name, value)

    # 훈련 관련 정보 저장
    for val_name, value in info['training_args'].items():
        value = check_bool(value)
        setattr(training_args, val_name, value)

    # 커스텀 옵션 정보 저장
    for val_name, value in info['custom_args'].items():
        value = check_bool(value)
        setattr(custom_args, val_name, value)
    
    return model_args, data_args, training_args, custom_args

def inference_config_setting(
    config
):
    '''
        ODQA Inference 기본 값 이외 변수 추가
    '''

    info = get_info(config.config_file_path)
    model_args = DataArguments()
    model_args, data_args, dense_args, training_args = \
        ModelArguments(), DataArguments(), DenseTrainingArguments(), TrainingArguments(config.inference_name)
    
    # 모델 관련 정보 저장
    for val_name, value in info['model_args'].items():
        value = check_bool(value)
        setattr(model_args, val_name, value)

    # 데이터 관련 정보 저장
    for val_name, value in info['data_args'].items():
        value = check_bool(value)
        setattr(data_args, val_name, value)

    # 훈련 관련 정보 저장
    for val_name, value in info['training_args'].items():
        value = check_bool(value)
        setattr(training_args, val_name, value)

    # Dense Retrieval 옵션 정보 저장
    for val_name, value in info['dense_args'].items():
        value = check_bool(value)
        setattr(dense_args, val_name, value)
    
    return model_args, data_args, dense_args, training_args

def wandb_config_setting(model_args, training_args, custom_args):
    '''
    WandB info에서 더 보고 싶은 정보 변수로 추가
    E.g.
    config.num_of_hidden_layer = 5
    config.seed = training_args.seed
    '''
    
    config = wandb.config

    # 모델 관련 정보 저장
    config.base_model = model_args.model_name_or_path
    config.epochs = training_args.num_train_epochs
    config.learning_rate = training_args.learning_rate
    config.train_batch_size = training_args.per_device_train_batch_size
    config.valid_batch_size = training_args.per_device_eval_batch_size
    config.accumulation_step = training_args.gradient_accumulation_steps

    # Description 저장
    config.description = custom_args.description
    
    return config


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="models/roberta-cnn",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default="models/roberta-cnn",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="models/roberta-cnn",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./data/train_dataset",
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
        default="Sparse",
        metadata={"help": "Kind of retrieval."},
    )

    top_k_retrieval: int = field(
        default=35,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
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
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use Wandb"})
    entity_name: str = field(
        default="clue",
        metadata={"help": "Your entity name in WandB E.g. clue or KyunghyunLim, ..."},
    )
    project_name: str = field(
        default="mrc_test",
        metadata={"help": "Your project name in WandB E.g. LKH, Readers, ..."},
    )
    wandb_run_name: str = field(
        default="roberta-cnn-512_v0.1",
        metadata={
            "help": "run name in WandB E.g. Bart_v0.1, Roberta_v0.1, DPR_Bert_v0.1"
        },
    )
    description: str = field(
        default="강민님이 만든 데이터 영끌",
        metadata={"help": "Explain your specific experiments settings"},
    )

    


@dataclass
class DenseTrainingArguments:
    """
    Arguments for training dense retrieval
    """
    data_path: str = field(
        default="./data/train_dataset",
        metadata={
            "help": "Train data path"
        },
    )
    dense_base_model: str = field(
        default="klue/roberta-small",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    dense_mode: str = field(
        default="single", #double
        metadata={
            "help": "single: share weight between p_encoder, q_encoder / double: not share"
        },
    )
    dense_passage_retrieval_name: str = field(
        default="./models/best/p_encoder",
        metadata={
            "help": "Path to pretrained model"
        },

    )
    dense_question_retrieval_name: str = field(
        default="./models/best/q_encoder",
        metadata={
            "help": "Path to pretrained model"
        },
    )
    dense_train_epoch: int = field(
        default=10,
        metadata={
            "help": "Epochs"
        },
    )
    dense_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "batch size for train DataLoader"
        },
    )
    dense_train_learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": "learning_rate for training"
        },
    )
    dense_context_max_length: int = field(
        default=384,
        metadata={
            "help": "batch size for train DataLoader"
        },
    )
    dense_question_max_length: int = field(
        default=80,
        metadata={
            "help": "batch size for train DataLoader"
        },
    )
    dense_train_output_dir: str = field(
        default="./models_result/roberta_small_dense_retireval_v4/",
        metadata={
            "help": "save directory"
        },
    )
    use_wiki_data: bool = field(
        default=True,
        metadata={
            "help": "Whether to use wiki data or not."
        },
    )
    wiki_data_path: str = field(
        default="./opt/ml/git/mrc-level2-nlp-13/data/wiki",
        metadata={
            "help": "Wiki data path"
        },
    )

@dataclass
class QuestionGenerationArguments:
    """
    Arguments for Question Generation
    """
    
    wiki_data_path: Optional[str] = field(
        default="../data/wikipedia_documents.json",
        metadata={"help": "exact path for wiki json file"},
    )
    qg_batch_size: int = field(
        default=128,
        metadata={"help": "batch size for QG"},
    )

def config_setting_for_dense_retrieval(
    data_args: DataArguments,
    custom_args: CustomArguments):
    '''
    WandB info에서 더 보고 싶은 정보 변수로 추가
    E.g.
    config.num_of_hidden_layer = 5
    config.seed = training_args.seed
    '''
    config = wandb.config
    config.dense_base_model = data_args.dense_base_model
    config.dense_passage_retrieval_name = data_args.dense_passage_retrieval_name
    config.dense_question_retrieval_name = data_args.dense_question_retrieval_name
    config.dense_train_epoch = data_args.dense_train_epoch
    config.dense_train_batch_size = data_args.dense_train_batch_size
    config.dense_train_learning_rate = data_args.dense_train_learning_rate
    config.dense_train_output_dir = data_args.dense_train_output_dir
    
    return config