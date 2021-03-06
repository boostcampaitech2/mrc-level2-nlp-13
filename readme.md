# KLUE Machine Reading Comprehension 
- π Naver Boost camp AI tech 2nd , Team CLUE 
- π [Wrap-up report](https://jonhyuk0922.notion.site/_level2_-MRC_13-7bdc5e677ff84b29ad52003f473fe625) , ποΈ [presentation slide](https://docs.google.com/presentation/d/1vLpOJDt0f3Rpaq3w6BbKD4yMfvlWzpE_1852N1NBt50/edit?usp=sharing)

## 1.Project Abstract

β KLUE MRC(Machine Reading Comprehension) DatasetμΌλ‘ μ£Όμ΄μ§ μ§λ¬Έμ λν λ¬Έμ κ²μ ν λ΅λ³ μΆμΆνλ Task.

β Retriver λ₯Ό ν΅ν΄  wikipediaμμ Top-k λ¬Έμλ₯Ό λΆλ¬μ€κ³ , Readerλ₯Ό ν΅ν΄ λ¬Έμ λ΄ λ΅λ³μ μΆμΆνλ λͺ¨λΈμ κ΅¬μΆ, μ€ν νμ¬ μ£Όμ΄μ§ μ§λ¬Έμ μ νν λ΅λ³μ μ°Ύμλ΄λ λͺ¨λΈμ λ§λλ κ².

β 1μΌ ν μ μΆνμλ 10νλ‘ μ νλμμ΅λλ€.


## 2. μ€μΉ λ°©λ²

π [dataset λ€μ΄λ‘λ](https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz)

```
# data (51.2 MB)
tar -xzf data.tar.gz
```

π ν΄λΉ λ ν¬ λ€μ΄λ‘λ
```
git clone https://github.com/boostcampaitech2/mrc-level2-nlp-13.git
```

π Poetryλ₯Ό ν΅ν ν¨ν€μ§ λ²μ  κ΄λ¦¬ 

```
# curl μ€μΉ
apt-get install curl #7.58.0

# poetry μ€μΉ
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# poetry ν­μμ± νμ±ν
~/.bashrcλ₯Ό μμ νμ¬ poetryλ₯Ό shellμμ μ¬μ© ν  μ μλλ‘ κ°μνκ²½μ μΆκ°
poetry use [μ¬μ©νλ κ°μνκ²½μ `python path` | κ°μνκ²½μ΄ μ€νμ€μ΄λΌλ©΄ `python`]  

# repo download ν λ²μ  μ μ© (poetry.tomlμ λ°λΌ μ μ©)
poetry install
```


## 3. ποΈ νλ‘μ νΈ κ΅¬μ‘°
### 3-1. μ μ₯μ κ΅¬μ‘°
```
mrc-level2-nlp-13
βββ configs
β   βββ example.json
βββ model
β   βββ Reader
β   β   βββ RobertaCnn.py
β   β   βββ trainer_qa.py
β   βββ Retrieval
β       βββ retrieval.py
βββ inference.py
βββ notebook
β   βββ post_preprocessing.ipynb
βββ ensemble
β   βββ hard_vote.ipynb
βββ augmentation
β   βββ quesiton_generate.py
βββ images
β   βββ dataset.png
βββ poetry.lock
βββ pyproject.toml
βββ readme.md
βββ License.md
βββ dense_retrieval_train.py
βββ train_reader.py
βββ utils
    βββ arguments.py
    βββ dense_utils
    β   βββ retrieval_dataset.py
    β   βββ utils.py
    βββ logger.py
    βββ utils_qa.py
```
### 3-2.λ°μ΄ν° κ΅¬μ‘° 

μλλ μ κ³΅νλ λ°μ΄ν°μμ λΆν¬λ₯Ό λ³΄μ¬μ€λλ€.

![λ°μ΄ν° λΆν¬](./images/dataset.png)

λ°μ΄ν°μμ νΈμμ±μ μν΄ Huggingface μμ μ κ³΅νλ datasetsλ₯Ό μ΄μ©νμ¬ pyarrow νμμ λ°μ΄ν°λ‘ μ μ₯λμ΄μμ΅λλ€. λ€μμ λ°μ΄ν°μμ κ΅¬μ±μλλ€.

```python
./data/                        # μ μ²΄ λ°μ΄ν°
    ./train_dataset/           # νμ΅μ μ¬μ©ν  λ°μ΄ν°μ. train κ³Ό validation μΌλ‘ κ΅¬μ± 
    ./test_dataset/            # μ μΆμ μ¬μ©λ  λ°μ΄ν°μ. validation μΌλ‘ κ΅¬μ± 
    ./wikipedia_documents.json # μν€νΌλμ λ¬Έμ μ§ν©. retrievalμ μν΄ μ°μ΄λ corpus.
```
λ§μ½ λ°μ΄ν° μ¦κ°μ ν΅ν datasetμ μ¬μ©νμ λ€λ©΄, μ΄ λλ ν λ¦¬μ μΆκ°ν΄μ£Όμκ³ 
config λ΄ "data_args" λ₯Ό λ³κ²½ν΄μ£Όμλ©΄ λ©λλ€.


## 4. train, evaluation , inference
### 4-1. π train

roberta λͺ¨λΈμ μ¬μ©ν  κ²½μ°, token type idsλ₯Ό μ¬μ©μνλ―λ‘ tokenizer μ¬μ©μ μλ ν¨μμ μ΅μμ μμ ν΄μΌν©λλ€.
λ² μ΄μ€λΌμΈμ klue/bert-baseλ‘ μ§νλλ μ΄ λΆλΆμ μ£Όμμ ν΄μ νμ¬ μ¬μ©ν΄μ£ΌμΈμ ! 
tokenizerλ train, validation (train.py), test(inference.py) μ μ²λ¦¬λ₯Ό μν΄ νΈμΆλμ΄ μ¬μ©λ©λλ€.
(tokenizerμ return_token_type_ids=Falseλ‘ μ€μ ν΄μ£Όμ΄μΌ ν¨)
- νμ΅μ νμν νλΌλ―Έν°λ₯Ό configs directory λ°μ .json νμΌλ‘ μμ±νμ¬ μ€νμ μ§νν©λλ€.
- νμ΅λ λͺ¨λΈμ tuned_models/"model_name" directoryμ bin fileμ ννλ‘ μ μ₯λ©λλ€.
```
# train_reader.py
def prepare_train_features(examples):
        # truncationκ³Ό padding(lengthκ° μ§§μλλ§)μ ν΅ν΄ toknizationμ μ§ννλ©°, strideλ₯Ό μ΄μ©νμ¬ overflowλ₯Ό μ μ§ν©λλ€.
        # κ° exampleλ€μ μ΄μ μ contextμ μ‘°κΈμ© κ²ΉμΉκ²λ©λλ€.
        tokenized_examples = tokenizer(
            ... ...
            #return_token_type_ids=False, # robertaλͺ¨λΈμ μ¬μ©ν  κ²½μ° False, bertλ₯Ό μ¬μ©ν  κ²½μ° Trueλ‘ νκΈ°ν΄μΌν©λλ€.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

```
# train_reader argparser
-c, --config_file_path : train config μ λ³΄κ° λ€μ΄μλ json fileμ μ΄λ¦
-l ,--log_file_path : train loggingμ ν  νμΌ μ΄λ¦
-n ,--model_name : λͺ¨λΈμ΄ μ μ₯λ  λλ ν λ¦¬ μ΄λ¦
--do_train : Readerλͺ¨λΈ train flag
--do_eval : Readerλͺ¨λΈ validation flag
```

- reader νμ΅ μμ
```
python train_reader.py -c ./configs/exp1.json -l exp1.log -n experiments1 --do_train
```
    

- dense retriver νμ΅ μμ
```
python train_reader.py -c ./configs/dense_exp1.json -l dense_exp1.log -n dense_experiment1 --do_train
```

### 4-2. π eval

MRC λͺ¨λΈμ μ±λ₯ νκ°(κ²μ¦)λ (`--do_eval`) νλ κ·Έλ₯Ό λ°λ‘ μ€μ ν΄μΌ ν©λλ€.  μ νμ΅ μμμ λ¨μν `--do_eval` μ μΆκ°λ‘ μλ ₯ν΄μ νλ ¨ λ° νκ°λ₯Ό λμμ μ§νν  μλ μμ΅λλ€.

```
# mrc λͺ¨λΈ νκ° (train/validation μ¬μ©)
python train_reader.py -c ./configs/exp1.json -l exp1.log -n experiments1 --do_train --do_eval
```

### 4-3. π₯ inference

retrieval κ³Ό mrc λͺ¨λΈμ νμ΅μ΄ μλ£λλ©΄ `inference.py` λ₯Ό μ΄μ©ν΄ odqa λ₯Ό μ§νν  μ μμ΅λλ€.

* νμ΅ν λͺ¨λΈμ  test_datasetμ λν κ²°κ³Όλ₯Ό μ μΆνκΈ° μν΄μ  μΆλ‘ (`--do_predict`)λ§ μ§ννλ©΄ λ©λλ€. 

* νμ΅ν λͺ¨λΈμ΄ train_dataset λν΄μ ODQA μ±λ₯μ΄ μ΄λ»κ² λμ€λμ§ μκ³  μΆλ€λ©΄ νκ°(--do_eval)λ₯Ό μ§ννλ©΄ λ©λλ€.

```
# ODQA μ€ν (test_dataset μ¬μ©)
# wandb κ° λ‘κ·ΈμΈ λμ΄μλ€λ©΄ μλμΌλ‘ κ²°κ³Όκ° wandb μ μ μ₯λ©λλ€. μλλ©΄ λ¨μν μΆλ ₯λ©λλ€
# inference argparser
-c, --config_file_path : inference config μ λ³΄κ° λ€μ΄μλ json fileμ μ΄λ¦
-l ,--log_file_path : inference loggingμ ν  νμΌ μ΄λ¦
-n ,--inference_name : inference κ²°κ³Όκ° μ μ₯λ  λλ ν λ¦¬ μ΄λ¦
-m , --model_name_or_path : inferenceμ μ¬μ©ν  λͺ¨λΈ λλ ν λ¦¬μ μ΄λ¦
```

```
python inference.py -c infer1.json -l infer1.log --n infer1_result -m ./tuned_models/train_dataset/ --do_predict
```

### 4-4. How to submit
`inference.py` νμΌμ μ μμμ²λΌ `--do_predict` μΌλ‘ μ€ννλ©΄ `--inference_name` μμΉμ `predictions.json` μ΄λΌλ νμΌμ΄ μμ±λ©λλ€. ν΄λΉ νμΌμ μ μΆν΄μ£Όμλ©΄ λ©λλ€.

### 4-5. MRC λͺ¨λΈ νμ΅ κ²°κ³Ό
λ€μμ MRC λͺ¨λΈμ public & private datsetμ λν κ²°κ³Όλ₯Ό λ³΄μ¬μ€λλ€.

- Public 19ν μ€ 9λ± π₯
![Public π₯](./images/public.png)

- Private 19ν μ€ 7λ± π₯
![Private π₯](./images/private.png)


## 5. Things to know

1. `inference.py` μμ TF-IDF scoreμ κ²½μ° sparse embedding μ νλ ¨νκ³  μ μ₯νλ κ³Όμ μ μκ°μ΄ μ€λ κ±Έλ¦¬μ§ μμ λ°λ‘ argument μ default κ° Trueλ‘ μ€μ λμ΄ μμ΅λλ€. μ€ν ν sparse_embedding.bin κ³Ό tfidfv.bin μ΄ μ μ₯μ΄ λ©λλ€. **λ§μ½ sparse retrieval κ΄λ ¨ μ½λλ₯Ό μμ νλ€λ©΄, κΌ­ λ νμΌμ μ§μ°κ³  λ€μ μ€νν΄μ£ΌμΈμ!** μκ·Έλ¬λ©΄ μ‘΄μ¬νλ νμΌμ΄ load λ©λλ€.
2. λͺ¨λΈμ κ²½μ° `--overwrite_cache` λ₯Ό μΆκ°νμ§ μμΌλ©΄ κ°μ ν΄λμ μ μ₯λμ§ μμ΅λλ€. 

3. ./predictions/ ν΄λ λν `--overwrite_output_dir` μ μΆκ°νμ§ μμΌλ©΄ κ°μ ν΄λμ μ μ₯λμ§ μμ΅λλ€.


## 6. License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
