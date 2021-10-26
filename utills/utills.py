import wandb
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
    training_args.overwrite_output_dir = custom_args.overwite
    config.epochs = training_args.num_train_epochs = custom_args.epochs
    config.learning_rate = training_args.learning_rate = custom_args.custom_learning_rate
    config.train_batch_size = training_args.per_device_train_batch_size = custom_args.train_batch_size
    config.valid_batch_size = training_args.per_device_eval_batch_size = custom_args.valid_batch_size
    config.accumulation_step = training_args.gradient_accumulation_steps = custom_args.accumulation_step
    
    print(training_args.num_train_epochs)
    
    return config

def logging_console(question_context, predictions, ground_truth):
    print('Q and Context: ')
    print(question_context)
    print('Prediction: ')
    print(predictions)
    print('Answer: ')
    print(ground_truth)

def logging_txt_file(f, question_context, predictions, ground_truth):
    f.write('-'*100 + '\n')
    f.write('Q and Context:\n')
    f.write(question_context + '\n')
    f.write('Prediction\n')
    f.write(predictions + '\n')
    f.write('Answer\n')
    f.write(ground_truth + '\n')
    f.write('-'*100 + '\n')
    