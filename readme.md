# KLUE Machine Reading Comprehension 
- 🍀 Naver Boost camp AI tech 2nd , Team CLUE 
- 📎 [Wrap-up report](https://jonhyuk0922.notion.site/_level2_-MRC_13-7bdc5e677ff84b29ad52003f473fe625) , 🖇️ [presentation slide](https://docs.google.com/presentation/d/1vLpOJDt0f3Rpaq3w6BbKD4yMfvlWzpE_1852N1NBt50/edit?usp=sharing)

## 1.Project Abstract

✋ KLUE MRC(Machine Reading Comprehension) Dataset으로 주어진 질문에 대한 문서 검색 후 답변 추출하는 Task.

✋ Retriver 를 통해  wikipedia에서 Top-k 문서를 불러오고, Reader를 통해 문서 내 답변은 추출한다.

✋ 1일 팀 제출횟수는 10회로 제한되었습니다.


## 2. 설치 방법

👉 [dataset 다운로드](https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz)

```
# data (51.2 MB)
tar -xzf data.tar.gz
```

👉 해당 레포 다운로드
```
git clone https://github.com/boostcampaitech2/mrc-level2-nlp-13.git
```

👉 Poetry를 통한 패키지 버전 관리 

```
# curl 설치
apt-get install curl #7.58.0

# poetry 설치
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# poetry 탭완성 활성화
poetry help completions  

# repo download 후 버전 적용 (poetry.toml에 따라 적용)
poetry update
```


## 3. 🏗️ 프로젝트 구조
### 3-1. 저장소 구조
```
mrc-level2-nlp-13
├── configs
│   └── example.json
├── model
│   ├── Reader
│   │   ├── RobertaCnn.py
│   │   └── trainer_qa.py
│   └── Retrieval
│       └── retrieval.py
├── inference.py
├── notebook
│   └── post_preprocessing.ipynb
├── ensemble
│   └── hard_vote.ipynb
├── augmentation
│   └── quesiton_generate.py
├── images
│   └── dataset.png
├── poetry.lock
├── pyproject.toml
├── readme.md
├── dense_retrieval_train.py
├── train_reader.py
└── utils
    ├── arguments.py
    ├── dense_utils
    │   ├── retrieval_dataset.py
    │   └── utils.py
    ├── logger.py
    └── utils_qa.py
```
### 3-2.데이터 구조 

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./images/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```python
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```
만약 데이터 증강을 통한 dataset을 사용하신다면, 이 디렉토리에 추가해주시고
config 내 "data_args" 를 변경해주시면 됩니다.


## 4. train, evaluation , inference
### 4-1. 🚆 train

roberta 모델을 사용할 경우, token type ids를 사용안하므로 tokenizer 사용시 아래 함수의 옵션을 수정해야합니다.
베이스라인은 klue/bert-base로 진행되니 이 부분의 주석을 해제하여 사용해주세요 ! 
tokenizer는 train, validation (train.py), test(inference.py) 전처리를 위해 호출되어 사용됩니다.
(tokenizer의 return_token_type_ids=False로 설정해주어야 함)

```
# train_reader.py
def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            ... ...
            #return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

```
# train_reader argparser
-c, --config_file_path : train config 정보가 들어있는 json file의 이름
-l ,--log_file_path : train logging을 할 파일 이름
-n ,--model_name : 모델이 저장될 디렉토리 이름
--do_train : Reader모델 train flag
--do_eval : Reader모델 validation flag
```

- reader 학습 예시 (train_dataset 사용)
```
python train_reader.py --c ./configs/example.json --l logs/ --n ./tuned_models/train_dataset/ --do_train
```

- dense retriver 학습 예시 (train_dataset 사용)
```
python dense_retrieval_train.py --c ./configs/example.json --l logs/ --n ./tuned_models/train_dataset/ --do_train
```

### 4-2. 📜 eval

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.  위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```
# mrc 모델 평가 (train_dataset 사용)
python train_reader.py --c ./configs/example.json --l logs/ --n ./tuned_models/train_dataset/ --do_train
```

### 4-3. 🥕 inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

* 학습한 모델의  test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 

* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(--do_eval)를 진행하면 됩니다.

```
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
# inference argparser
-c, --config_file_path : inference config 정보가 들어있는 json file의 이름
-l ,--log_file_path : inference logging을 할 파일 이름
-n ,--inference_name : inference 결과가 저장될 디렉토리 이름
-m , --model_name_or_path : inference에 사용할 모델 디렉토리의 이름
```

```
python inference.py --c ./configs/example.json --l logs/ --n ./predictions/ -m ./tuned_models/train_dataset/ 
```

### How to submit
`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--inference_name` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

### MRC 모델 학습 결과
다음은 MRC 모델의 public & private datset에 대한 결과를 보여줍니다.

- Public 19팀 중 9등 🥈
![Public 🥈](./images/public.png)

- Private 19팀 중 7등 🥈
![Private 🥈](./images/private.png)

## Things to know

1. `inference.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 존재하는 파일이 load 됩니다.
2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 

3. ./predictions/ 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.
