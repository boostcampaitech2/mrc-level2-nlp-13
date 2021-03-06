
##################
# Import modules #
##################

from typing import *
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import load_from_disk

from torch.utils.data import Dataset
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from utils.arguments import DenseTrainingArguments

#######################
# Classes & Functions #
#######################

def prepare_train_features_for_retriever(
    examples: Dataset, 
    tokenizer, 
    question_column_name: str,
    context_column_name: str,
    answer_column_name: str,
    max_seq_length: int
    ):
    '''
        Arguments:
            examples Dataset:
                torch.utill.Dataset으로 이루어진 Context, Questions, Answer를 받습니다.
            tokenizer 
                context를 tokenizing 하기 위한 tokenizer
            question_column_name:
                Dataset에서 Question을 닮고 있는 변수 이름 'question'
            context_column_name:
                Dataset에서 Context 닮고 있는 변수 이름 'context'
            answer_column_name:
                Dataset에서 Answer를 닮고 있는 변수 이름 'answer'
            max_seq_length:
                context를 tokenizing할 때, 사용할 최대 길이
        Returns:
            Dict -> ['input_ids', 'attention_mask', 'label', ...]

        Note:
            context를 받아 토크나이징 하고, 정답이 있는 strid인지 아닌지 
    '''
    pad_on_right = tokenizer.padding_side == "right"
    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples[context_column_name if pad_on_right else question_column_name],
        truncation=True,
        max_length=max_seq_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length" #if False else False,
    )

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
    # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["labels"] = []
    tokenized_examples['sample_mapping'] = sample_mapping
    
    for i, offsets in enumerate(tqdm(offset_mapping)):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        if len(answers["answer_start"]) == 0:
            tokenized_examples["labels"].append(1)
        else:
            # text에서 정답의 Start/end character index
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # text에서 current span의 Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != 0:
                token_start_index += 1

            # text에서 current span의 End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["labels"].append(1)
            else:
                tokenized_examples["labels"].append(0)
    return tokenized_examples

def prepare_data(tokenizer, dense_args: DenseTrainingArguments):
    '''
        Arguments:
            tokenizer 
                context 및 qustion를 tokenizing 하기 위한 tokenizer
            dense_args:
                dense 학습 설정을 위한 arguments 들
        Returns:
            DataLoader(), List[List[str]], List[List[str]]

        Note:
            데이터를 로드하고, 토크나이징해 학습에 사용하기 위한 dataloader, list로 반환합니다.
    '''
    datasets = load_from_disk(dense_args.data_path)
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]

    if dense_args.use_wiki_data:
        wiki_datasets = load_from_disk(dense_args.wiki_data_path)
        q_seqs = tokenizer(list(np.concatenate([datasets['train']['question'], wiki_datasets['question']], axis=0)), max_length=dense_args.dense_question_max_length, padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(list(np.concatenate([datasets['train']['context'], wiki_datasets['context']], axis=0)), max_length=dense_args.dense_context_max_length, padding="max_length", truncation=True, return_tensors='pt')
    else:
        q_seqs = tokenizer(train_dataset['question'], max_length=dense_args.dense_question_max_length, padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(train_dataset['context'], max_length=dense_args.dense_context_max_length, padding="max_length", truncation=True, return_tensors='pt')

    # Train 데이터 준비
    train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                        q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=dense_args.dense_train_batch_size)

    # Valid 데이터 준비
    valid_corpus = list([example['context'] for example in valid_dataset])

    return train_dataloader, valid_corpus, valid_dataset['question']