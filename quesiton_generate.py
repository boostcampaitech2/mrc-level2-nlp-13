import os, sys, getopt
import json
import pandas as pd
import random
from tqdm import tqdm
from pororo import Pororo
from arguments import QuestionGenerationArguments
from transformers import HfArgumentParser
import torch

def create_additional_datasets(mode):
    """
        Create additional dataset with wikipedia data
        Arguments:
            mode:
                (default): with title
                'ner': with ner
        Return:
            None
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    data_path  = "../data/"
    context_path =  "wikipedia_documents.json"

    with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    wiki_df = pd.DataFrame.from_dict(wiki, orient='index')
    wiki_df['document_id'] = 'wiki-'+ wiki_df['document_id'].astype("string")

    contexts = list(wiki_df['text'])
    doc_id = list(wiki_df['document_id'])

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
                answers.append('')
    else:
        answers = list(wiki_df['title'])

    full_length = wiki_df.shape[0] 
    batch_size = 128
    batch_num = int(full_length/batch_size)
    st = 0
    end = batch_size
    wiki_qg_df = pd.DataFrame()
    value_dic = {
                    'doc_id' :[],
                    'context' : [],
                    'answer' : [],
                    'question' : []
                    }

    for bn in tqdm(range(batch_num+1)):
        try:
            batch_answers = answers[st:end]
            batch_contexts = contexts[st:end]
            batch_doc_id = doc_id[st:end]

            qg = Pororo(task="qg", lang="ko")
            questions= qg(batch_answers,batch_contexts)

            if bn % 10 == 0:
                print(batch_answers[:5])
                print('*'*30)
                print(batch_contexts[:5])
                print('*'*30)
                print(questions[:5])
                print('*'*30)
                print(len(value_dic['question']))

            value_dic['doc_id'].extend(batch_doc_id)
            value_dic['context'].extend(batch_contexts)
            value_dic['answer'].extend(batch_answers)
            value_dic['question'].extend(questions)

            del qg, questions


        except IndexError:
            print('index error up please check index from start : '+str(st))
            
        st = end
        # for last batch
        if bn == batch_num - 1:
            end = full_length
        else:
            end = st + batch_size

        torch.cuda.empty_cache()

    wiki_qg_df.append(value_dic, ignore_index=True)
    wiki_qg_df.to_csv('./wiki_qg.csv',index=False)

def main(argv):
    file_name = argv[0] # 실행시키는 파일명
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
    create_additional_datasets(run_mode)

if __name__ == "__main__":
    main(sys.argv)
