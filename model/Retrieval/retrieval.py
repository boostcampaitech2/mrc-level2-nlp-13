##################
# Import modules #
##################

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import re
import time

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from utils.dense_utils.retrieval_dataset import *
from utils.logger import get_logger
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader 
from rank_bm25 import BM25Okapi

from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch, helpers

########################
# Set global variables #
########################
logger = get_logger("logs/retrieval.log")

#####################
# Class & Functions #
#####################

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.3f} s")

class RetrievalBasic:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        # Path 설정 및 데이터 로드
        self.data_path = data_path
        logger.info(type(data_path),data_path)
        logger.info(type(context_path),context_path)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # Context 정렬 및 index 할당
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        logger.info(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Set tokenizer
        self.tokenizer = tokenize_fn
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다.
        self.indexer = None  # build_faiss()로 생성합니다.

def make_elastic_data():
    with open('./data/wikipedia_documents.json', 'r') as f:
        wiki_json = json.load(f)
        wiki_df = pd.DataFrame(wiki_json).transpose()

    wiki_data = wiki_df.drop_duplicates(['text']) # 3876
    wiki_data = wiki_df.reset_index(drop=True)

    wiki_data['text_origin'] = wiki_data['text']

    wiki_data['text_origin'] = wiki_data['text_origin'].apply(
        lambda x : ' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔァ-ヴー々〆〤一-龥~₩!@#$%^&*()“”‘’《》≪≫〈〉『』「」＜＞_+|{}:"<>?`\-=\\[\];',.\/·]''', 
        ' ', 
        str(x.lower().strip())).split())
    )

    wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n\\n',' '))
    wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n\n',' '))
    wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n',' '))
    wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n',' '))

    wiki_data['text'] = wiki_data['text'].apply(
        lambda x : ' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9~₩!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]''', 
        ' ', 
        str(x.lower().strip())).split())
    )

    title = []
    text = []
    text_origin = []

    for num in tqdm(range(len(wiki_data))):
        title.append(wiki_data['title'][num])
        text.append(wiki_data['text'][num])
        text_origin.append(wiki_data['text_origin'][num])

    df = pd.DataFrame({
        'title':title,
        'text':text,
        'text_origin':text_origin
    })

    return df    


class DenseRetrieval(RetrievalBasic):
    def __init__(self, tokenizers, encoders, data_path: Optional[str] = "./data/", context_path: Optional[str] = "wikipedia_documents.json") -> NoReturn:
        super().__init__(None, data_path=data_path, context_path=context_path)
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            encoders:
                passage_encoder와 question_encoder를 의미합니다. (passage_encoder, question_encoder)

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 DenseRetrieval를 이용해 임베딩 벡터를 생성하는 기능을 합니다.
        """
        self.p_tokenizer = tokenizers[0]
        self.q_tokenizer = tokenizers[1]
        self.p_encoder = encoders[0]
        self.q_encoder = encoders[1]
        self.passage_embedding_vectors = []

    def get_dense_passage_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들어 self에 저장 
        """
        logger.info('Tokenize passage')
        item = self.p_tokenizer(self.contexts, max_length=500, padding="max_length", truncation=True, return_tensors='pt')
        p_dataset = RetrievalValidDataset(input_ids=item['input_ids'], attention_mask=item['attention_mask'])
        p_loader = DataLoader(p_dataset, batch_size=16)

        logger.info('Make passage embedding vectors')
        for item in tqdm(p_loader):
            self.passage_embedding_vectors.extend(
                    self.p_encoder(input_ids = item['input_ids'].to('cuda:0'), attention_mask=item['attention_mask'].to('cuda:0')).pooler_output.to('cpu').detach().numpy())
            torch.cuda.empty_cache()
            del item
        self.passage_embedding_vectors = torch.Tensor(self.passage_embedding_vectors).squeeze()
        logger.info('passage embedding vectors: ', self.passage_embedding_vectors.size())

    def retrieve(
        self, q_encoder, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        assert self.passage_embedding_vectors is not None, "get_dense_passage_embedding() 메소드를 먼저 수행해줘야합니다."
        
    
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(q_encoder, query_or_dataset, k=topk)
            logger.info("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                logger.info(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                logger.info(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(q_encoder, 
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, q_encoder, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        q_seqs_val = self.q_tokenizer([query], max_length=80, padding="max_length", truncation=True, return_tensors='pt').to('cuda')

        logger.info('Make top-k passage per query')
        q_emb = q_encoder(**q_seqs_val).pooler_output.detach().cpu()  #(num_query, emb_dim)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.passage_embedding_vectors, 0, 1))

        rank = torch.argsort(dot_prod_scores, descending=True)
       
        doc_score = dot_prod_scores[rank[:k]]
        doc_indices = rank[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, q_encoder, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        logger.info('Get passage per each question')
        q_seqs_val = self.q_tokenizer(queries, max_length=80, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        q_dataset = RetrievalValidDataset(input_ids=q_seqs_val['input_ids'], attention_mask=q_seqs_val['attention_mask'])
        q_loader = DataLoader(q_dataset, batch_size=1)
        
        doc_scores = []
        doc_indices = []
        for item in tqdm(q_loader):
            q_embs = q_encoder(input_ids = item['input_ids'].to('cuda:0'), attention_mask=item['attention_mask'].to('cuda:0')).pooler_output.to('cpu')  #(num_query, emb_dim)
            for q_emb in q_embs:
                dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.passage_embedding_vectors, 0, 1))
                rank = torch.argsort(dot_prod_scores, dim=0, descending=True).squeeze()
            
                doc_scores.append(dot_prod_scores[rank[:k]].detach().cpu().numpy())
                doc_indices.append(rank[:k].detach().cpu().numpy())
        return doc_scores, doc_indices

class ElasticSearch():
    def __init__(self):
        self.es_server = Popen(['../elastic/elasticsearch-7.9.2/bin/elasticsearch'],
                   stdout=PIPE, 
                   stderr=STDOUT,
                   preexec_fn = lambda: os.setuid(1)
                  )
        time.sleep(10)
        self.es = Elasticsearch('localhost:9200')
    
    def set_elasticsearch(self):

        """
        Summary:
            Elastic Search에 활용할 기본틀을 만들고
            local 서버로 wiki 데이터를 저장합니다
        """
        if self.es.indices.exists('document'):
            self.es.indices.delete(index='document')

        self.es.indices.create(index = 'document',
            body = {
                'settings':{
                    'analysis':{
                        'analyzer':{
                            'my_analyzer':{
                                "type": "custom",
                                'tokenizer':'nori_tokenizer',
                                'decompound_mode':'mixed',
                                'stopwords':'_korean_',
                                'synonyms':'_korean_',
                                "filter": ["lowercase",
                                            "my_shingle_f",
                                            "nori_readingform",
                                            "nori_number",
                                            "cjk_bigram",
                                            "decimal_digit",
                                            "stemmer",
                                            "trim"]
                            }
                        },
                        'filter':{
                            'my_shingle_f':{
                                "type": "shingle"
                            }
                        }
                    },
                    'similarity':{
                        'my_similarity':{
                            'type':'BM25',
                        }
                    }
                },
                'mappings':{
                    'properties':{
                        'title':{
                            'type':'text',
                            'analyzer':'my_analyzer',
                            'similarity':'my_similarity'
                        },
                        'text':{
                            'type':'text',
                            'analyzer':'my_analyzer',
                            'similarity':'my_similarity'
                        },
                        'text_origin':{
                            'type':'text',
                            'analyzer':'my_analyzer',
                            'similarity':'my_similarity'
                        }
                    }
                }
            }
        )

        df = make_elastic_data()
        buffer = []
        rows = 0

        for num in tqdm(range(len(df))):
            article = {"_id": num,
                    "_index": "document", 
                    "title" : df['title'][num],
                    "text" : df['text'][num],
                    "text_origin" : df['text_origin'][num]}
            buffer.append(article)
            rows += 1
            if rows % 3000 == 0:
                helpers.bulk(self.es, buffer)
                buffer = []
                logger.info("Inserted {} articles".format(rows), end="\r")
                time.sleep(1)

        if buffer:
            helpers.bulk(self.es, buffer)

    def get_elasticsearch(self):
        return self.es

class SparseRetrieval(RetrievalBasic):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        embedding_form : Optional[str] = "TF-IDF"
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            embedding_form:
                Sparse embedding의 함수를 결정합니다 TF-IDF, BM25, ES 중 고를 수 있습니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        # Set Basic variables
        super().__init__(tokenize_fn, data_path, context_path)
        self.embedding_form = embedding_form

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.tokenizer,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.bm25 = None


    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            Embedding_form에 따른 Passage Embedding을 만들어 줍니다.
            
            TF-IDF :
                Embedding을 pickle로 저장합니다.
                만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
            BM25 :
                BM250kpi Class를 생성하여 embedding 합니다.
            ES :
                ElasticSerch Class를 생성하여 embedding 합니다.
        """
        # 파일 이름 및 경로 설정
        pickle_name = f"sparse_embedding_{self.embedding_form}.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        # Pickle을 저장합니다.
        if self.embedding_form == "TF-IDF":
            
            tfidfv_name = f"tfidv.bin"
            tfidfv_path = os.path.join(self.data_path, tfidfv_name)

            if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                with open(tfidfv_path, "rb") as file:
                    self.tfidfv = pickle.load(file)
                logger.info("Embedding pickle load.")
            else:
                logger.info(f"Build passage embedding for {self.embedding_form}")
                self.p_embedding = self.tfidfv.fit_transform(self.contexts)
                logger.info(self.p_embedding.shape)
                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(tfidfv_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                logger.info("Embedding pickle saved.")

        elif self.embedding_form == "BM25":
            logger.info("Tokenize Text for BM25 Object")
            tokenized_contexts = list(map(self.tokenizer, tqdm(self.contexts)))

            logger.info("Allocate BM25 Object")
            self.bm25 = BM25Okapi(tqdm(tokenized_contexts))

        elif self.embedding_form == "ES":
            logger.info("Start Elastic Search")
            es = ElasticSearch()
            logger.info("Set Elastic Search")
            es.set_elasticsearch()
            logger.info("Finish Elastic Search")
            self.es = es

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                
                str 형태
                - `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태 
                - query를 포함한 HF.Dataset을 받습니다.
                - 이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.

            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        
        if self.embedding_form == "TF-IDF":
            assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        
        elif self.embedding_form in ["BM25","ES"]:
            pass
            
        # 단순 Query 하나일 경우
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            logger.info("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                logger.info(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                logger.info(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        # Query의 묶음일 경우
        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )

            for idx, example in enumerate(tqdm(query_or_dataset)):
                
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],

                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Return:
            Tuple[doc_score(List), doc_indices(List)]
            
            doc_score : Query에 대한 k개의 문서 유사도 점수
            doc_indices : Query에 대한 k개의 문서 인덱스
        """

        if self.embedding_form == "TF-IDF":

            with timer("transform"):
                query_vec = self.tfidfv.transform([query])
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            with timer("query ex search"):
                result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            result = result.squeeze()

        elif self.embedding_form == "BM25":
            logger.info("----- Start Calculate BM25 -----")
            logger.info("----- Tokenize querys -----")
            tokenized_query = self.tokenizer(query)
            logger.info("----- get scores from querys -----")
            result = self.bm25.get_scores(tokenized_query)      

        elif self.embedding_form == "ES":
            logger.info("----- Start Calculate Elastic Search -----")
            result = self.es.es.search(index = "document", q=query, size=k)
            doc_score = [hit['_score'] for hit in result['hits']['hits']]
            doc_indices = [hit['_id'] for hit in result['hits']['hits']]
            logger.info("----- Finish Calculate Elastic Search -----")

            return doc_score, doc_indices

        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (List):
                여러개의 Query를 받습니다.
            k (Optional[int]): 1
                query 당 상위 몇 개의 Passage를 반환할지 정합니다.
        Return:
            Tuple[doc_score(List), doc_indices(List)]
            
            doc_score : Querys에 대한 k개의 문서 유사도 점수
            doc_indices : Querys에 대한 k개의 문서 인덱스
        """
        if self.embedding_form == "TF-IDF":
            query_vec = self.tfidfv.transform(queries)
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            result = query_vec * self.p_embedding.T
            
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            

        elif self.embedding_form == "BM25":
            logger.info("----- Start Calculate BM25 -----")
            logger.info("----- Tokenize querys -----")
            tokenized_queries = list(map(self.tokenizer, tqdm(queries)))
            logger.info("----- get scores from querys -----")
            result = np.array(list(map(self.bm25.get_scores, tqdm(tokenized_queries))))

        elif self.embedding_form == "ES":
            logger.info("----- Start Calculate Elastic Search -----")
            
            doc_score = []
            doc_indices = []

            for query in tqdm(queries):
                res = self.es.es.search(index = "document",q=query, size=k)
                doc_score.append([hit['_score'] for hit in res['hits']['hits']])
                doc_indices.append([int(hit['_id']) for hit in res['hits']['hits']])
            logger.info("----- Finish Calculate Elastic Search -----")

            return doc_score, doc_indices

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices


class JointRetrieval(RetrievalBasic):
    def __init__(
        self,
        sparse_tokenizer,
        dense_tokenizer,
        encoders,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        embedding_form : Optional[str] = "BM25"
    ) -> NoReturn:
        
        """
        Arguments:
            sparse_tokenizer:
                Sparse Retrieval Score를 구할 때 들어가는 토크나이저 입니다.
                
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            dense_tokenizer:
                Dense Retrieval Score를 구할 때 들어가는 토크나이저 입니다.

                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        # Set Basic variables
        super().__init__(sparse_tokenizer, data_path, context_path)

        self.sparse = SparseRetrieval(
            sparse_tokenizer,
            data_path,
            context_path,
            embedding_form
        )

        self.dense = DenseRetrieval(
            dense_tokenizer,
            encoders,
            data_path,
            context_path
        )

        self.sparse.get_sparse_embedding()
        self.dense.get_dense_passage_embedding()


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
    
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            logger.info("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                logger.info(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                logger.info(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], bm_k=300, dense_k=100
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk(
        self, queries: List, bm_k: Optional[int] = 1, dense_k: Optional[int] = 1,
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        

        _, doc_indices = self.sparse.get_relevant_doc_bulk(queries, bm_k)

        dense_tokenized_queries = self.dense.q_tokenizer(
            queries, 
            max_length=80, 
            padding="max_length", 
            truncation=True, 
            return_tensors='pt'
        ).to('cuda')

        passage_embedding_vectors = np.array(self.dense.passage_embedding_vectors)
        logger.info(type(passage_embedding_vectors))
        q_dataset = RetrievalValidDataset(input_ids=dense_tokenized_queries['input_ids'], attention_mask=dense_tokenized_queries['attention_mask'])
        q_loader = DataLoader(q_dataset, batch_size=1)

        dense_doc_scores = []
        dense_doc_indices = []
        for item,indices in tqdm(zip(q_loader,doc_indices)):
            q_embs = self.dense.q_encoder(input_ids = item['input_ids'].to('cuda:0'), attention_mask=item['attention_mask'].to('cuda:0')).pooler_output.to('cpu')

            for q_emb in q_embs:
                mapping_indices = np.array(indices)
                dot_prod_scores = torch.matmul(q_emb, torch.transpose(torch.tensor(passage_embedding_vectors[indices]), 0, 1))
                rank = torch.argsort(dot_prod_scores, dim=0, descending=True).squeeze()
            
                dense_doc_scores.append(dot_prod_scores[rank[:dense_k]].detach().cpu().numpy())
                dense_doc_indices.append(mapping_indices[rank[:dense_k].detach().cpu().numpy()])

        return dense_doc_scores, dense_doc_indices

