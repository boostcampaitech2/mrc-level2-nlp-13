import os
import json
import time
import datasets
import faiss
import pickle
import numpy as np
import pandas as pd
import re
import time

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from retrieval_module.retrieval_dataset import *

import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from scipy import sparse as sp
from datasets import Dataset, load_from_disk, concatenate_datasets, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch, helpers


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class RetrievalBasic:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        # Path �ㅼ젙 諛� �곗씠�� 濡쒕뱶
        self.data_path = data_path
        print(type(data_path), data_path)
        print(type(context_path), context_path)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # Context �뺣젹 諛� index �좊떦
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set �� 留ㅻ쾲 �쒖꽌媛� 諛붾�뚮�濡�
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Set tokenizer
        self.tokenizer = tokenize_fn
        self.p_embedding = None  # get_sparse_embedding()濡� �앹꽦�⑸땲��.
        self.indexer = None  # build_faiss()濡� �앹꽦�⑸땲��.


def make_elastic_data():
    with open("../data/wikipedia_documents.json", "r") as f:
        wiki_data = pd.DataFrame(json.load(f)).transpose()

    wiki_data = wiki_data.drop_duplicates(["text"])  # 3876
    wiki_data = wiki_data.reset_index()
    del wiki_data["index"]

    wiki_data["text_origin"] = wiki_data["text"]

    wiki_data["text_origin"] = wiki_data["text_origin"].apply(
        lambda x: " ".join(
            re.sub(
                r"""[^ \r\n��-�롢뀖-�ｊ�-�즑-zA-Z0-9��-�붵궊-�담꺖�끹�녴�ㅴ�-榕�~��!@#$%^&*()�쒋�앪�섃�쇻�듽�뗢돦�ャ�덀�됥�롢�뤵�뚣�랃폒竊�_+|{}:"<>?`\-=\\[\];',.\/쨌]""",
                " ",
                str(x.lower().strip()),
            ).split()
        )
    )

    wiki_data["text"] = wiki_data["text"].apply(lambda x: x.replace("\\n\\n", " "))
    wiki_data["text"] = wiki_data["text"].apply(lambda x: x.replace("\n\n", " "))
    wiki_data["text"] = wiki_data["text"].apply(lambda x: x.replace("\\n", " "))
    wiki_data["text"] = wiki_data["text"].apply(lambda x: x.replace("\n", " "))

    wiki_data["text"] = wiki_data["text"].apply(
        lambda x: " ".join(
            re.sub(
                r"""[^ \r\n��-�롢뀖-�ｊ�-�즑-zA-Z0-9~��!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]""",
                " ",
                str(x.lower().strip()),
            ).split()
        )
    )

    title = []
    text = []
    text_origin = []

    for num in tqdm(range(len(wiki_data))):
        title.append(wiki_data["title"][num])
        text.append(wiki_data["text"][num])
        text_origin.append(wiki_data["text_origin"][num])

    df = pd.DataFrame({"title": title, "text": text, "text_origin": text_origin})
    return df


class DenseRetrieval(RetrievalBasic):
    def __init__(
        self,
        tokenizers,
        encoders,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        super().__init__(None, data_path=data_path, context_path=context_path)
        """
        Arguments:
            tokenize_fn:
                湲곕낯 text瑜� tokenize�댁＜�� �⑥닔�낅땲��.
                �꾨옒�� 媛숈� �⑥닔�ㅼ쓣 �ъ슜�� �� �덉뒿�덈떎.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag�� Mecab

            data_path:
                �곗씠�곌� 蹂닿��섏뼱 �덈뒗 寃쎈줈�낅땲��.

            context_path:
                Passage�ㅼ씠 臾띠뿬�덈뒗 �뚯씪紐낆엯�덈떎.

            encoders:
                passage_encoder�� question_encoder瑜� �섎��⑸땲��. (passage_encoder, question_encoder)

            data_path/context_path媛� 議댁옱�댁빞�⑸땲��.

        Summary:
            Passage �뚯씪�� 遺덈윭�ㅺ퀬 DenseRetrieval瑜� �댁슜�� �꾨쿋�� 踰≫꽣瑜� �앹꽦�섎뒗 湲곕뒫�� �⑸땲��.
        """
        self.p_tokenizer = tokenizers[0]
        self.q_tokenizer = tokenizers[1]
        self.p_encoder = encoders
        self.passage_embedding_vectors = []

    def get_dense_passage_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding�� 留뚮뱾�� self�� ����
        """
        print("Tokenize passage")
        item = self.p_tokenizer(
            self.contexts,
            max_length=500,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_dataset = RetrievalValidDataset(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        p_loader = DataLoader(p_dataset, batch_size=16)

        print("Make passage embedding vectors")
        for item in tqdm(p_loader):
            self.passage_embedding_vectors.extend(
                self.p_encoder(
                    input_ids=item["input_ids"].to("cuda:0"),
                    attention_mask=item["attention_mask"].to("cuda:0"),
                )
                .pooler_output.to("cpu")
                .detach()
                .numpy()
            )
            torch.cuda.empty_cache()
            del item
        self.passage_embedding_vectors = torch.Tensor(
            self.passage_embedding_vectors
        ).squeeze()
        print("passage embedding vectors: ", self.passage_embedding_vectors.size())

    def retrieve(
        self, q_encoder, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str�대굹 Dataset�쇰줈 �대（�댁쭊 Query瑜� 諛쏆뒿�덈떎.
                str �뺥깭�� �섎굹�� query留� 諛쏆쑝硫� `get_relevant_doc`�� �듯빐 �좎궗�꾨� 援ы빀�덈떎.
                Dataset �뺥깭�� query瑜� �ы븿�� HF.Dataset�� 諛쏆뒿�덈떎.
                �� 寃쎌슦 `get_relevant_doc_bulk`瑜� �듯빐 �좎궗�꾨� 援ы빀�덈떎.
            topk (Optional[int], optional): Defaults to 1.
                �곸쐞 紐� 媛쒖쓽 passage瑜� �ъ슜�� 寃껋씤吏� 吏��뺥빀�덈떎.

        Returns:
            1媛쒖쓽 Query瑜� 諛쏅뒗 寃쎌슦  -> Tuple(List, List)
            �ㅼ닔�� Query瑜� 諛쏅뒗 寃쎌슦 -> pd.DataFrame: [description]

        Note:
            �ㅼ닔�� Query瑜� 諛쏅뒗 寃쎌슦,
                Ground Truth媛� �덈뒗 Query (train/valid) -> 湲곗〈 Ground Truth Passage瑜� 媛숈씠 諛섑솚�⑸땲��.
                Ground Truth媛� �녿뒗 Query (test) -> Retrieval�� Passage留� 諛섑솚�⑸땲��.
        """
        assert (
            self.passage_embedding_vectors is not None
        ), "get_dense_passage_embedding() 硫붿냼�쒕� 癒쇱� �섑뻾�댁쨾�쇳빀�덈떎."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(
                q_encoder, query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve�� Passage瑜� pd.DataFrame�쇰줈 諛섑솚�⑸땲��.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    q_encoder, query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query�� �대떦 id瑜� 諛섑솚�⑸땲��.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve�� Passage�� id, context瑜� 諛섑솚�⑸땲��.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation �곗씠�곕� �ъ슜�섎㈃ ground_truth context�� answer�� 諛섑솚�⑸땲��.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(
        self, q_encoder, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                �섎굹�� Query瑜� 諛쏆뒿�덈떎.
            k (Optional[int]): 1
                �곸쐞 紐� 媛쒖쓽 Passage瑜� 諛섑솚�좎� �뺥빀�덈떎.
        Note:
            vocab �� �녿뒗 �댁긽�� �⑥뼱濡� query �섎뒗 寃쎌슦 assertion 諛쒖깮 (��) �ｋ솂?
        """
        q_seqs_val = self.q_tokenizer(
            [query],
            max_length=80,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        print("Make top-k passage per query")
        q_emb = (
            q_encoder(**q_seqs_val).pooler_output.detach().cpu()
        )  # (num_query, emb_dim)
        dot_prod_scores = torch.matmul(
            q_emb, torch.transpose(self.passage_embedding_vectors, 0, 1)
        )

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
                �섎굹�� Query瑜� 諛쏆뒿�덈떎.
            k (Optional[int]): 1
                �곸쐞 紐� 媛쒖쓽 Passage瑜� 諛섑솚�좎� �뺥빀�덈떎.
        Note:
            vocab �� �녿뒗 �댁긽�� �⑥뼱濡� query �섎뒗 寃쎌슦 assertion 諛쒖깮 (��) �ｋ솂?
        """
        print("Get passage per each question")
        q_seqs_val = self.q_tokenizer(
            queries,
            max_length=80,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        q_dataset = RetrievalValidDataset(
            input_ids=q_seqs_val["input_ids"],
            attention_mask=q_seqs_val["attention_mask"],
        )
        q_loader = DataLoader(q_dataset, batch_size=1)

        doc_scores = []
        doc_indices = []
        for item in tqdm(q_loader):
            q_embs = q_encoder(
                input_ids=item["input_ids"].to("cuda:0"),
                attention_mask=item["attention_mask"].to("cuda:0"),
            ).pooler_output.to(
                "cpu"
            )  # (num_query, emb_dim)
            for q_emb in q_embs:
                dot_prod_scores = torch.matmul(
                    q_emb, torch.transpose(self.passage_embedding_vectors, 0, 1)
                )
                rank = torch.argsort(dot_prod_scores, dim=0, descending=True).squeeze()

                doc_scores.append(dot_prod_scores[rank[:k]].detach().cpu().numpy())
                doc_indices.append(rank[:k].detach().cpu().numpy())
        return doc_scores, doc_indices


class SparseRetrieval(RetrievalBasic):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        embedding_form: Optional[str] = "TF-IDF",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                湲곕낯 text瑜� tokenize�댁＜�� �⑥닔�낅땲��.
                �꾨옒�� 媛숈� �⑥닔�ㅼ쓣 �ъ슜�� �� �덉뒿�덈떎.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag�� Mecab

            data_path:
                �곗씠�곌� 蹂닿��섏뼱 �덈뒗 寃쎈줈�낅땲��.

            context_path:
                Passage�ㅼ씠 臾띠뿬�덈뒗 �뚯씪紐낆엯�덈떎.

            embedding_form:
                Sparse embedding�� �⑥닔瑜� 寃곗젙�⑸땲�� TF-IDF, BM25 以� 怨좊� �� �덉뒿�덈떎.

            data_path/context_path媛� 議댁옱�댁빞�⑸땲��.

        Summary:
            Passage �뚯씪�� 遺덈윭�ㅺ퀬 TfidfVectorizer瑜� �좎뼵�섎뒗 湲곕뒫�� �⑸땲��.
        """
        # Set Basic variables
        super().__init__(tokenize_fn, data_path, context_path)

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.tokenizer,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.bm25 = None

        # Set default variables
        self.embedding_type = {
            "TF-IDF": self.tfidfv,
            "BM25": self.bm25,
        }
        self.embedding_form = embedding_form

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding�� 留뚮뱾怨�
            Embedding�� pickle濡� ���ν빀�덈떎.
            留뚯빟 誘몃━ ���λ맂 �뚯씪�� �덉쑝硫� ���λ맂 pickle�� 遺덈윭�듬땲��.
        """
        # �뚯씪 �대쫫 諛� 寃쎈줈 �ㅼ젙
        pickle_name = f"sparse_embedding_{self.embedding_form}.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        # Pickle�� ���ν빀�덈떎.
        if self.embedding_form == "TF-IDF":

            tfidfv_name = f"tfidv.bin"
            tfidfv_path = os.path.join(self.data_path, tfidfv_name)

            if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                with open(tfidfv_path, "rb") as file:
                    self.tfidfv = pickle.load(file)
                print("Embedding pickle load.")
            else:
                print(f"Build passage embedding for {self.embedding_form}")
                self.p_embedding = self.tfidfv.fit_transform(self.contexts)
                print(self.p_embedding.shape)
                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(tfidfv_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                print("Embedding pickle saved.")

        elif self.embedding_form == "BM25":
            print("Allocate BM25 Object")
            tokenized_contexts = list(map(self.tokenizer, self.contexts))
            self.bm25 = BM25Okapi(tqdm(tokenized_contexts))

        elif self.embedding_form == "ES":
            print("Start Elastic Search")

            es_server = Popen(
                ["../elastic/elasticsearch-7.9.2/bin/elasticsearch"],
                stdout=PIPE,
                stderr=STDOUT,
                preexec_fn=lambda: os.setuid(1),
            )
            time.sleep(30)
            es = Elasticsearch("localhost:9200")
            if es.indices.exists("document"):
                es.indices.delete(index="document")
            es.indices.create(
                index="document",
                body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "my_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "decompound_mode": "mixed",
                                    "stopwords": "_korean_",
                                    "synonyms": "_korean_",
                                    "filter": [
                                        "lowercase",
                                        "my_shingle_f",
                                        "nori_readingform",
                                        "nori_number",
                                        "cjk_bigram",
                                        "decimal_digit",
                                        "stemmer",
                                        "trim",
                                    ],
                                }
                            },
                            "filter": {"my_shingle_f": {"type": "shingle"}},
                        },
                        "similarity": {
                            "my_similarity": {
                                "type": "BM25",
                                #   'type':'boolean',
                            }
                        },
                    },
                    "mappings": {
                        "properties": {
                            "title": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity",
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity",
                            },
                            "text_origin": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity",
                            },
                        }
                    },
                },
            )

            df = make_elastic_data()
            buffer = []
            rows = 0

            for num in tqdm(range(len(df))):
                article = {
                    "_id": num,
                    "_index": "document",
                    "title": df["title"][num],
                    "text": df["text"][num],
                    "text_origin": df["text_origin"][num],
                }
                buffer.append(article)
                rows += 1
                if rows % 3000 == 0:
                    helpers.bulk(es, buffer)
                    buffer = []
                    print("Inserted {} articles".format(rows), end="\r")
                    time.sleep(1)

            if buffer:
                helpers.bulk(es, buffer)

            self.es = es
            print("Total articles inserted: {}".format(rows))

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            �띿꽦�쇰줈 ���λ릺�� �덈뒗 Passage Embedding��
            Faiss indexer�� fitting �쒖폒�볦뒿�덈떎.
            �대젃寃� ���λ맂 indexer�� `get_relevant_doc`�먯꽌 �좎궗�꾨� 怨꾩궛�섎뒗�� �ъ슜�⑸땲��.

        Note:
            Faiss�� Build�섎뒗�� �쒓컙�� �ㅻ옒 嫄몃━湲� �뚮Ц��,
            留ㅻ쾲 �덈∼寃� build�섎뒗 寃껋� 鍮꾪슚�⑥쟻�낅땲��.
            洹몃젃湲� �뚮Ц�� build�� index �뚯씪�� ���뺥븯怨� �ㅼ쓬�� �ъ슜�� �� 遺덈윭�듬땲��.
            �ㅻ쭔 �� index �뚯씪�� �⑸웾�� 1.4Gb+ �닿린 �뚮Ц�� �щ윭 num_clusters濡� �쒗뿕�대낫怨�
            �쒖씪 �곸젅�� 寃껋쓣 �쒖쇅�섍퀬 紐⑤몢 ��젣�섎뒗 寃껋쓣 沅뚯옣�⑸땲��.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str�대굹 Dataset�쇰줈 �대（�댁쭊 Query瑜� 諛쏆뒿�덈떎.
                str �뺥깭�� �섎굹�� query留� 諛쏆쑝硫� `get_relevant_doc`�� �듯빐 �좎궗�꾨� 援ы빀�덈떎.
                Dataset �뺥깭�� query瑜� �ы븿�� HF.Dataset�� 諛쏆뒿�덈떎.
                �� 寃쎌슦 `get_relevant_doc_bulk`瑜� �듯빐 �좎궗�꾨� 援ы빀�덈떎.
            topk (Optional[int], optional): Defaults to 1.
                �곸쐞 紐� 媛쒖쓽 passage瑜� �ъ슜�� 寃껋씤吏� 吏��뺥빀�덈떎.

        Returns:
            1媛쒖쓽 Query瑜� 諛쏅뒗 寃쎌슦  -> Tuple(List, List)
            �ㅼ닔�� Query瑜� 諛쏅뒗 寃쎌슦 -> pd.DataFrame: [description]

        Note:
            �ㅼ닔�� Query瑜� 諛쏅뒗 寃쎌슦,
                Ground Truth媛� �덈뒗 Query (train/valid) -> 湲곗〈 Ground Truth Passage瑜� 媛숈씠 諛섑솚�⑸땲��.
                Ground Truth媛� �녿뒗 Query (test) -> Retrieval�� Passage留� 諛섑솚�⑸땲��.
        """
        if self.embedding_form == "TF-IDF":
            assert (
                self.p_embedding is not None
            ), "get_sparse_embedding() 硫붿냼�쒕� 癒쇱� �섑뻾�댁쨾�쇳빀�덈떎."
        elif self.embedding_form == "BM25":
            pass

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve�� Passage瑜� pd.DataFrame�쇰줈 諛섑솚�⑸땲��.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query�� �대떦 id瑜� 諛섑솚�⑸땲��.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve�� Passage�� id, context瑜� 諛섑솚�⑸땲��.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation �곗씠�곕� �ъ슜�섎㈃ ground_truth context�� answer�� 諛섑솚�⑸땲��.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                �섎굹�� Query瑜� 諛쏆뒿�덈떎.
            k (Optional[int]): 1
                �곸쐞 紐� 媛쒖쓽 Passage瑜� 諛섑솚�좎� �뺥빀�덈떎.
        Note:
            vocab �� �녿뒗 �댁긽�� �⑥뼱濡� query �섎뒗 寃쎌슦 assertion 諛쒖깮 (��) �ｋ솂?
        """
        if self.embedding_form == "TF-IDF":
            with timer("transform"):
                query_vec = self.tfidfv.transform([query])
            assert (
                np.sum(query_vec) != 0
            ), "�ㅻ쪟媛� 諛쒖깮�덉뒿�덈떎. �� �ㅻ쪟�� 蹂댄넻 query�� vectorizer�� vocab�� �녿뒗 �⑥뼱留� 議댁옱�섎뒗 寃쎌슦 諛쒖깮�⑸땲��."

            with timer("query ex search"):
                result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()

            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_score = result.squeeze()[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]

            return doc_score, doc_indices

        elif self.embedding_form == "BM25":

            tokenized_query = self.tokenizer(query)
            result = self.bm25.get_scores(tokenized_query)

            sorted_result = np.argsort(result)[::-1]
            doc_score = result[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]

            return doc_score, doc_indices

        elif self.embedding_form == "ES":
            res = self.es.search(index="document", q=query, size=k)
            doc_score = [hit["_score"] for hit in res["hits"]["hits"]]
            doc_indices = [hit["_id"] for hit in res["hits"]["hits"]]
            return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                �섎굹�� Query瑜� 諛쏆뒿�덈떎.
            k (Optional[int]): 1
                �곸쐞 紐� 媛쒖쓽 Passage瑜� 諛섑솚�좎� �뺥빀�덈떎.
        Note:
            vocab �� �녿뒗 �댁긽�� �⑥뼱濡� query �섎뒗 寃쎌슦 assertion 諛쒖깮 (��) �ｋ솂?
        """
        if self.embedding_form == "TF-IDF":
            query_vec = self.tfidfv.transform(queries)
            assert (
                np.sum(query_vec) != 0
            ), "�ㅻ쪟媛� 諛쒖깮�덉뒿�덈떎. �� �ㅻ쪟�� 蹂댄넻 query�� vectorizer�� vocab�� �녿뒗 �⑥뼱留� 議댁옱�섎뒗 寃쎌슦 諛쒖깮�⑸땲��."

            result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices

        elif self.embedding_form == "BM25":
            print("----- Start Calculate BM25 -----")
            print("----- Tokenize querys -----")
            tokenized_queries = list(map(self.tokenizer, tqdm(queries)))
            print("----- get scores from querys -----")
            result = np.array(list(map(self.bm25.get_scores, tqdm(tokenized_queries))))

            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices

        elif self.embedding_form == "ES":
            print("----- Start Calculate Elastic Search -----")
            doc_score = []
            doc_indices = []
            for query in tqdm(queries):
                res = self.es.search(index="document", q=query, size=k)
                # print([hit['_score'] for hit in res['hits']['hits']])
                # print([int(hit['_id']) for hit in res['hits']['hits']])
                doc_score.append([hit["_score"] for hit in res["hits"]["hits"]])
                doc_indices.append([int(hit["_id"]) for hit in res["hits"]["hits"]])

            return doc_score, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str�대굹 Dataset�쇰줈 �대（�댁쭊 Query瑜� 諛쏆뒿�덈떎.
                str �뺥깭�� �섎굹�� query留� 諛쏆쑝硫� `get_relevant_doc`�� �듯빐 �좎궗�꾨� 援ы빀�덈떎.
                Dataset �뺥깭�� query瑜� �ы븿�� HF.Dataset�� 諛쏆뒿�덈떎.
                �� 寃쎌슦 `get_relevant_doc_bulk`瑜� �듯빐 �좎궗�꾨� 援ы빀�덈떎.
            topk (Optional[int], optional): Defaults to 1.
                �곸쐞 紐� 媛쒖쓽 passage瑜� �ъ슜�� 寃껋씤吏� 吏��뺥빀�덈떎.

        Returns:
            1媛쒖쓽 Query瑜� 諛쏅뒗 寃쎌슦  -> Tuple(List, List)
            �ㅼ닔�� Query瑜� 諛쏅뒗 寃쎌슦 -> pd.DataFrame: [description]

        Note:
            �ㅼ닔�� Query瑜� 諛쏅뒗 寃쎌슦,
                Ground Truth媛� �덈뒗 Query (train/valid) -> 湲곗〈 Ground Truth Passage瑜� 媛숈씠 諛섑솚�⑸땲��.
                Ground Truth媛� �녿뒗 Query (test) -> Retrieval�� Passage留� 諛섑솚�⑸땲��.
            retrieve�� 媛숈� 湲곕뒫�� �섏�留� faiss.indexer瑜� �ъ슜�⑸땲��.
        """

        assert self.indexer is not None, "build_faiss()瑜� 癒쇱� �섑뻾�댁＜�몄슂."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve�� Passage瑜� pd.DataFrame�쇰줈 諛섑솚�⑸땲��.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query�� �대떦 id瑜� 諛섑솚�⑸땲��.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve�� Passage�� id, context瑜� 諛섑솚�⑸땲��.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation �곗씠�곕� �ъ슜�섎㈃ ground_truth context�� answer�� 諛섑솚�⑸땲��.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                �섎굹�� Query瑜� 諛쏆뒿�덈떎.
            k (Optional[int]): 1
                �곸쐞 紐� 媛쒖쓽 Passage瑜� 諛섑솚�좎� �뺥빀�덈떎.
        Note:
            vocab �� �녿뒗 �댁긽�� �⑥뼱濡� query �섎뒗 寃쎌슦 assertion 諛쒖깮 (��) �ｋ솂?
        """

        query_vec = self.embedding_type[self.embedding_form].transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "�ㅻ쪟媛� 諛쒖깮�덉뒿�덈떎. �� �ㅻ쪟�� 蹂댄넻 query�� vectorizer�� vocab�� �녿뒗 �⑥뼱留� 議댁옱�섎뒗 寃쎌슦 諛쒖깮�⑸땲��."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                �섎굹�� Query瑜� 諛쏆뒿�덈떎.
            k (Optional[int]): 1
                �곸쐞 紐� 媛쒖쓽 Passage瑜� 諛섑솚�좎� �뺥빀�덈떎.
        Note:
            vocab �� �녿뒗 �댁긽�� �⑥뼱濡� query �섎뒗 寃쎌슦 assertion 諛쒖깮 (��) �ｋ솂?
        """

        query_vecs = self.embedding_type[self.embedding_form].transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "�ㅻ쪟媛� 諛쒖깮�덉뒿�덈떎. �� �ㅻ쪟�� 蹂댄넻 query�� vectorizer�� vocab�� �녿뒗 �⑥뼱留� 議댁옱�섎뒗 寃쎌슦 諛쒖깮�⑸땲��."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")

    args = parser.parse_args()

    config = BertConfig().from_pretrained("klue/bert-base")

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    p_encoder = BertModel(config=config)
    q_encoder = BertModel(config=config)
    # p_encoder = AutoModel.from_pretrained("klue/bert-base")
    # q_encoder = AutoModel.from_pretrained("klue/bert-base")

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
    )

    retriever = DenseRetrieval(
        tokenize_fn=tokenizer, args=args, p_encoder=p_encoder, q_encoder=q_encoder
    )
    train_dataset, eval_dataset = retriever.load_dataset(eval=True)
    # retriever.prepare_in_batch_negative(num_neg = 2)
    train_p_encoder, train_q_encoder = retriever.train(
        train_dataset=train_dataset, eval_dataset=eval_dataset, args=args
    )
    p_embedding = retriever.get_embedding(p_encoder=train_p_encoder)
    a, b = retriever.get_relevant_doc_bulk(
        p_embedding=p_embedding, q_encoder=train_q_encoder, topk=30
    )

    print("-" * 10, "question", "-" * 10)
    print(retriever.test[0])
    for i in b[0]:
        print("-" * 10, "test", str(i), "-" * 10)
        print(retriever.contexts[i])


#  [56717  9418 18690 20298 40171]
#  [37157 16361 56244 20579 56645]
#  [33898 25937 30465 43045 46471]
#  [29006 14121 46510 33594 28485]
#  [10103 45921 23183  1467 43159]
#  [54517 54329 56098 25904 52149]
#  [39502 27050 27044 28639 35692]
#  [38857 40171   590  7822 45952]
#  [10115 14459 40350 21292 54885]


# # Test sparse
# org_dataset = load_from_disk(args.dataset_name)
# full_ds = concatenate_datasets(
#     [
#         org_dataset["train"].flatten_indices(),
#         org_dataset["validation"].flatten_indices(),
#     ]
# )  # train dev 瑜� �⑹튇 4192 媛� 吏덈Ц�� ���� 紐⑤몢 �뚯뒪��
# print("*" * 40, "query dataset", "*" * 40)
# print(full_ds)

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#     args.model_name_or_path,
#     use_fast=False,
# )

# retriever = SparseRetrieval(
#     tokenize_fn=tokenizer.tokenize,
#     data_path=args.data_path,
#     context_path=args.context_path,
# )

# query = "���듬졊�� �ы븿�� 誘멸뎅�� �됱젙遺� 寃ъ젣沅뚯쓣 媛뽯뒗 援�� 湲곌���?"

# if args.use_faiss:

#     # test single query
#     with timer("single query by faiss"):
#         scores, indices = retriever.retrieve_faiss(query)

#     # test bulk
#     with timer("bulk query by exhaustive search"):
#         df = retriever.retrieve_faiss(full_ds)
#         df["correct"] = df["original_context"] == df["context"]

#         print("correct retrieval result by faiss", df["correct"].sum() / len(df))

# else:
#     with timer("bulk query by exhaustive search"):
#         df = retriever.retrieve(full_ds)
#         df["correct"] = df["original_context"] == df["context"]
#         print(
#             "correct retrieval result by exhaustive search",
#             df["correct"].sum() / len(df),
#         )

#     with timer("single query by exhaustive search"):
#         scores, indices = retriever.retrieve(query)
