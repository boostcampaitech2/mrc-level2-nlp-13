##################
# import modules #
##################

import os, sys, getopt
import json
import random
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

def load_data(wiki_data_path : str, run_mode : str = 'title'):
    """
    Arguments:
        wiki_data_path(str):
        - wiki json 파일의 경로입니다.
        - arugments.py의 wiki_data_path에서 가져올 예정입니다.
        run_mode(str):
        - answer로 활용할 데이터를 선택하는 파라미터입니다.
        - 기본값은 'title'이며, ner 태깅을 활용하려면 'ner'을 입력합니다.

    Returns:
        - wiki_df -> pd.DataFrame: [description]

    Summary:
        wiki 데이터를 불러와 pandas의 DataFrame 형태로 변환하는 기능을합니다.
    """
    
    with open(os.path.join(wiki_data_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    wiki_df = pd.DataFrame.from_dict(wiki, orient='index')
    wiki_df['document_id'] = 'wiki-'+ wiki_df['document_id'].astype("string")

    if mode == 'ner':
        ner = Pororo(task="ner", lang="ko")
        answers = []
        for context in contexts:
            try:
                tagged_ner = ner(context)
                tagged_ner = [(word, entity) for word, entity in tagged_ner if entity != 'O']
                answer = random.choice(tagged_ner)[0]
                answers.append(answer)
            except:
                answers.append('') # 추후 data 전처리로 제거.
    else:
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


def question_generation(run_mode: str):
    """
    Arguments:
        run_mode(str):
        - load_data() 함수에 필요한 파라미터
    Summary:
        wiki data에서 context와 title을 불러와 이를 pororo의 QG를 활용해 Question을 만들어낸 후 
        doc_id, context, answer, question이 담긴 csv파일을 저장한다.
    Note:
        만들어진 csv파일은 이후 pseudo labeling의 input으로 활용 됨.
    """
    parser = HfArgumentParser(QuestionGenerationArguments)
    wiki_data_path = parser.wiki_data_path
    answers, contexts, doc_id = load_data(wiki_data_path, run_mode)
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
    file_name = sys.argv[0] # 실행시키는 파일명
    run_mode = "title"  # running mode title(default) / ner

    try:
        print(argv[1:])
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "mode="])
    except getopt.GetoptError:
        print(file_name, "-c <mode>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-m <mode>")
            sys.exit(0)
        elif opt in ("-m", "--mode"):
            run_mode = arg

    if run_mode not in ['title', 'ner']:
        raise Exception('not expected args...')

    print(f"generate question with {run_mode}")
    question_generation(run_mode)
