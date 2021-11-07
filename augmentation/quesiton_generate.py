##################
# import modules #
##################

import os
import json
import pandas as pd
from tqdm import tqdm
from pororo import Pororo
from utils.arguments import QuestionGenerationArguments
from transformers import HfArgumentParser
import torch
from collections import defaultdict
from typing import Dict

os.environ["TOKENIZERS_PARALLELISM"] = "true"

#####################
# Class & Functions #
#####################

def load_data(wiki_data_path : str):
    """
    Arguments:
        wiki_data_path(str):
        - wiki json 파일의 경로입니다.
        - arugments.py의 wiki_data_path에서 가져올 예정입니다.

    Returns:
        - wiki_df -> pd.DataFrame: [description]

    Summary:
        wiki 데이터를 불러와 pandas의 DataFrame 형태로 변환하는 기능을합니다.
    """
    
    with open(os.path.join(wiki_data_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    wiki_df = pd.DataFrame.from_dict(wiki, orient='index')
    wiki_df['document_id'] = 'wiki-'+ wiki_df['document_id'].astype("string")

    answers = wiki_df['title'].to_list()
    contexts = wiki_df['text'].to_list()
    doc_id = wiki_df['document_id'].to_list()

    return answers, contexts, doc_id


def save_data(value_dic: Dict, output_path: str):
    """
    Arguments:
        value_dic (Dict):
            doc_id, context, answer, quetions 정보를 가지고 있는 딕셔너리를 받습니다.
        output_path (str):
            데이터가 저장되는 경로입니다.

    Summary:
        생성한 데이터를 csv형태로 저장하는 기능을 합니다.
    """
    wiki_qg_df = pd.DataFrame()
    wiki_qg_df.append(value_dic, ignore_index=True)
    wiki_qg_df.to_csv(output_path,index=False)


def question_generation():
    """
    Summary:
        wiki data에서 context와 title을 불러와 이를 pororo의 QG를 활용해 Question을 만들어낸 후 
        doc_id, context, answer, question이 담긴 csv파일을 저장한다.
    Note:
        만들어진 csv파일은 이후 pseudo labeling의 input으로 활용 됨.
    """
    parser = HfArgumentParser(QuestionGenerationArguments)
    wiki_data_path = parser.wiki_data_path
    answers, contexts, doc_id = load_data(wiki_data_path)
    full_length = len(contexts) 
    batch_size = parser.qg_batch_size
    batch_num = int(full_length/batch_size)
    start_index = 0
    end_index = batch_size
    value_dic = defaultdict(list)   
    
    for bn in tqdm(range(batch_num+1)):
        try:
            batch_answers = answers[start_index:end_index]
            batch_contexts = contexts[start_index:end_index]
            batch_doc_id = doc_id[start_index:end_index]

            qg = Pororo(task="qg", lang="ko")
            questions= qg(batch_answers,batch_contexts)

            value_dic['doc_id'].extend(batch_doc_id)
            value_dic['context'].extend(batch_contexts)
            value_dic['answer'].extend(batch_answers)
            value_dic['question'].extend(questions)

            del qg, questions

        except IndexError:
            print('index error up please check index from start : '+str(st))
            
        start_index = end_index
        # for last batch
        if bn == batch_num - 1:
            end_index = full_length
        else:
            end_index = start_index + batch_size

        torch.cuda.empty_cache()

    ## save result 함수
    save_data(value_dic,'./wiki_qg.csv')


if __name__ == "__main__":
    question_generation()