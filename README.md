# Pstage_03_데이터 제작(NLP)

Solution for Data Production(NLP) in 2nd BoostCamp AI Tech by **메타몽팀 (2조)**


## Content
- [Project Abstract](#project-abstract)
- [Result](#result)
- [Hardware](#hardware)
- [Operating System](#operating-system)
- [Archive Contents](#archive-contents)
- [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
  * [Install Requirements](#install-requirements)
- [Data Building Pipeline](#data-building-pipeline)
  * [Create topic-related dataset](#create-topic-related-dataset)
  * [Validation for Data Quality](#validation-for-data-quality)
  * [Train & Validation by BERT Model](#train---validation-by-bert-model)

## Project Abstract
- 스타트업 관련 위키피디아 원시 데이터로로 RE 데이터셋 구축을 통한 데이터셋 구축 프로세스 학습
    - 위키피디아 원시 데이터: 스타트업 관련 위키피디아 텍스트 파일 40개 (11번가, 린 스타트업, 무신사 등)
- 원시 데이터로부터 적합한 문장 및 Entity를 선정하고, subject-object entity의 관계 생성 및 검수를 통해 데이터 품질 평가(Fleiss's Kappa)


## Result

|Fleiss' Kappa|   Accuracy   |   F1   | Auprc |
|:------:|:------:|:------:|:------:|
| 0.799 | 0.747 | 72.227 |  79.850 |


## Hardware

- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Operating System

- Ubuntu 18.04.5 LTS

## Archive Contents

- data-annotation-nlp-level3-nlp-02 : 구현 코드와 모델 checkpoint 및 모델 결과를 포함하는 디렉토리

```
data-annotation-nlp-level3-nlp-02/
├── etc
│   ├── RE_relation_class_map.xlsx
│   ├── 스타트업_RE_task_가이드라인_v1.pdf
│   └── 스타트업_RE_task_가이드라인_v2.pdf
├── data
│   └── startup.csv
├── annotation.ipynb
├── calculate_iaa.py
├── dict_label_to_num.pkl
├── dict_num_to_label.pkl
├── evaluation.py
├── fleiss.py
├── handling_annotation.ipynb
├── inference.py
├── labeling.ipynb
├── load_data.py
├── requirements.sh
├── split_sentence_from_rawdata.ipynb
└── train.py
```

- `data/` : 스타트업 원시 데이터를 활용해 제작한 데이터가 있는 디렉토리
- `load_data` : data 디렉토리에 있는 .csv 파일을 불러오고 전처리 하는 파일
- `train.py` : relation extraction model 학습하고 validation score를 측정하는 파일
- `calculate_iaa.py` : 작업자의 data annotation 결과물을 iaa 지표로 측정하는 파일

## Getting Started

### Dependencies

- torch==1.6.0
- pandas==1.1.4
- transformers==4.11.0
- datasets==1.4.0
- gspread==4.0.1
- oauth2client==4.1.3

### Install Requirements

```
sh requirement_install.sh
```

## Data Building Pipeline

### Create topic-related dataset
1. 원시 데이터를 KSS spliter를 이용하여 문장 분리(1차 판별)
```
split_sentence_from_rawdata.ipynb
```

2. 정상 문장(2차 판별)을 [Tagtog](https://tagtog.net)에 입력 및 entity 선정
3. tagtog 데이터를 모델 평가를 위한 형태로 데이터 전처리
4. annotation 작업을 위하여 google sheet으로 데이터 전송 및 annotation 작업
5. 작업된 annotation 데이터에서 에러 데이터 및 오라벨 데이터 선별 후 데이터 수정 및 제거
```
handling_annotation.ipynb #3~5번 작업 코드
```
### Validation for Data Quality
데이터 품질 평가(IAA with Fleiss' Kappa)
```
python calculate_iaa.py
```

### Train & Validation by BERT Model
```
$ python train.py 
```
