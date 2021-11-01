import os
import wandb
import pandas as pd

from transformers import TrainingArguments 
from arguments import (
    ModelArguments,
    DataTrainingArguments,
    CustomArguments,
)

def config_setting(data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    custom_args: CustomArguments,):
    '''
    WandB info에서 더 보고 싶은 정보 변수로 추가
    E.g.
    config.num_of_hidden_layer = 5
    config.seed = training_args.seed
    '''
    
    config = wandb.config

    # 모델 관련 정보 저장
    config.base_model = model_args.model_name_or_path
    #config.retriver_model = developing ...

    # 훈련 관련 정보 설정 및 저장
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 1  # evaluation step.
    training_args.overwrite_output_dir = custom_args.overwite
    training_args.save_total_limit = 2
    config.epochs = training_args.num_train_epochs = custom_args.epochs
    config.learning_rate = training_args.learning_rate = custom_args.custom_learning_rate
    config.train_batch_size = training_args.per_device_train_batch_size = custom_args.train_batch_size
    config.valid_batch_size = training_args.per_device_eval_batch_size = custom_args.valid_batch_size
    config.accumulation_step = training_args.gradient_accumulation_steps = custom_args.accumulation_step
    
    config.description = custom_args.description

    print(training_args.num_train_epochs)
    
    return config

def logging_console(question_context, predictions, ground_truth):
    print('Q and Context: ')
    print(question_context)
    print('Prediction: ')
    print(predictions)
    print('Answer: ')
    print(ground_truth)

def logging_csv_file(file_name, question_context, predictions, ground_truth):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        new_df = pd.DataFrame()
        new_df['QnC'] = question_context
        new_df['Prediction'] = predictions
        new_df['ground_truth'] = ground_truth
        df = pd.concat([df,new_df], axis=0)
    else:
        df = pd.DataFrame()
        df['QnC'] = question_context
        df['Prediction'] = predictions
        df['ground_truth'] = ground_truth
    df.to_csv(file_name, index=False)
    