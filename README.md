# 1-1. 프로젝트 개요

## 1-1-1. 프로젝트 주제

  모델의 구조를 바꾸지 않고 오로지 학습 데이터를 가공하는 것만으로 모델의 성능을 끌어올릴 수 있다. 본 프로젝트에서는 topic classification task에서 모델과 학습 옵션들을 고정시키고 오로지 데이터의 증강 및 전처리만으로 모델 성능 향상에 도전한다.

## 1-1-2. 데이터셋과 모델

- klue/bert-base 모델을 사용하며 epoch=2, adam, weight_decay등의 하이퍼파라미터 고정하여 학습
- KLUE-YNAT 데이터셋
    - 학습 데이터 7,000개, 테스트 데이터 47,785개
    - Label : [0: "IT과학", 1: "경제",  2: "사회",  3: "생활문화", 4: "세계" ,  5:"스포츠" , 6: "정치"]
    - 학습데이터와 테스트 데이터 샘플에는 아래와 같이 noise가 추가되어 있음 (Graphemes to Phoneme)
    - 총 6852개의 noise가 반영된 데이터가 train/test 데이터에 들어가 있다. (12%)
        - G2P
        
        |  | 원래 문장 | 변형 결과 |
        | --- | --- | --- |
        | 소리나는대로 변경(규정) | 나의 친구는 계산이 아주 빠르다 | 나에 친구는 계사니 아주 빠르다 |
        | 소리나는대로 변경(사람들이 많이 말하는 방식) | 나의 친구는 계산이 아주 빠르다 | 나에 친구는 게사니 아주 빠르다 |
        | 숫자를 한글로 | 지금 시각은 12시 12분입니다 | 지금 시가그 녈두시 시비부님니다 |
        | 영어를 한글로 | 그 사람은 좀, old school 같아 | 그 사라믄 좀, 올드 스쿨 가타 |
        - Labeling Error
    
    ![Untitled](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/30d21519-0878-447e-b92b-e37cfcd4da14)


## 1-1-4. 평가 방법

- Macro F1 Score
    - 각 label 별 f1 score 를 구한 후 모든 label 의 각 f1 score 에 대한 평균을 구한다.

# 1-2. 프로젝트 팀 구성 및 역할

| 이름 | 역할 |
| --- | --- |
| 김용림 | 뉴스 크롤링 데이터 수집, 오픈소스 데이터 수집 |
| 송영우 | annotation 툴 제작, 오픈소스 데이터 수집, 데이터 클리닝, 데이터 버저닝 및 비교 |
| 이동근 | 모델 공유 허브 구축, inference 속도 향상, Streamlit 앱 제작 |
| 윤석원 | 데이터 EDA, 오픈소스데이터 수집, 데이터셋 조합에 따른 학습, 잘못된 labeling 분석, 데이터셋 공유 허브 구축 |
| 한혜민 | Confusion Matrix, 데이터 노이즈 추가(G2P), 뉴스 날짜별 크롤링, 데이터 번역 |

# 1-3. 프로젝트 수행 절차

## 

![screenshot-www notion so-2024 02 02-10_36_13](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/c99d8903-c465-4847-b02a-d06551497d21)

# 1-4. 프로젝트 수행 결과

### 1-4-1. 데이터셋 labeling 전수 검사
- 데이터셋에서 labeling의 오류를 사람이 직접 검사하고자 annotation tool 개발
![Untitled](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/f50d603d-75a9-4fda-856d-884856214d17)

### 1-4-2. 오픈소스 데이터 수집

- 네이버 뉴스 크롤링
    - 저작권을 존중해 AI 에 학습에 사용할 수 있는 데이터만 수집
    - 스포츠 분야를 제외한 정치, 경제, 사회, 생활/문화, IT/과학 분야의 기사 제목과 본문의 일부, 출처를 밝히기 위한 url 수집
    - 일부 특수문자와 한글, 영어, 한자만 남도록 data cleaning 수행
- kaggle news dataset (l[ink](https://www.kaggle.com/datasets/rmisra/news-category-dataset?resource=download))
    
    ```jsx
    {"link": "https://www.huffpost.com/entry/funniest-tweets-cats-dogs-september-17-23_n_632de332e4b0695c1d81dc02", 
    "headline": "23 Of The Funniest Tweets About Cats And Dogs This Week (Sept. 17-23)", 
    "category": "COMEDY", 
    "short_description": "\"Until you have a dog you don't understand what could be eaten.\"", 
    "authors": "Elyse Wanshel", 
    "date": "2022-09-23"}
    ```
    
    - 캐글에 공개된 영어 뉴스 데이터셋으로 사회 분야를 제외하고 [SPORTS, POLITICS, SCIENCE(IT과학), WORLD NEWS(세계), BUSINESS(경제), HOME & LIVING(생활문화)]에 해당하는 headline만 수집
    - facebook/mbart-large-50-many-to-many-mmt 모델로 한국어 번역하여 데이터셋 구축
- KETI-AIR/kor_ag_news ([link](https://huggingface.co/datasets/KETI-AIR/kor_ag_news))
    
    <img width="811" alt="Screenshot 2024-01-29 at 4 15 44 PM" src="https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/d46ac5c0-b21a-4107-a045-a63b7e304f75">

    - hugging-face에 있는 Agnews의 한국어 번역 데이터셋
    - label별로 31900개의 샘플들로 구성
- ai_hub 뉴스 MRC 데이터셋 ([link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=89))
    - 기계독해를 위한 뉴스 기사 데이터셋으로 기사 제목과 카테고리 정보만을 수집하여 구축
    - 약 11만개의 샘플들로 구성
- 모두의 말뭉치 ([국립국어원](https://kli.korean.go.kr/corpus/main/requestMain.do#down))
    - 신문 말뭉치, 신문 말뭉치2020, 신문 말뭉치2021, 신문 말뭉치2020 중 일부 사용
    
    <img width="1128" alt="스크린샷 2024-02-02 오전 10 29 39" src="https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/331de1f2-33ea-43f7-adc7-c4772c0cedb4">


### 1-4-3. 데이터 전처리

데이터 전처리는 크게 5가지 방법을 선택적으로 사용하였다.

- dot 특수문자 제거
    - dot 특수문자에 해당하는 '…', '...', '·’가 별다른 의미를 갖지 않으면서도, 서로 다른 토큰으로 인식되어 모델의 주제 분석에 혼동을 주고 있다고 가정하였다.
    - 따라서 이들 토큰을 전부 공백으로 치환한 뒤, strip()을 적용하여 불필요한 공백 또한 삭제하였다.
    - 결과: **f1 score 상승 0.8392 → 0.8432**
- number masking
    - text에 많은 숫자들이 존재하지만, 각각의 토큰이 ‘숫자’라는 추상적인 의미 외에 서로 다른 토큰으로 입력되고 있다는 점이 모델의 주제 분석에 혼동을 주고 있다고 가정하였다.
    - 모든 숫자를 ‘0’으로 masking하여 ‘숫자’라는 정보만 남기고, 각각의 값이 지니는 의미를 최소화하려고 시도하였다. (예: 2018 → 0000, 54.5 → 00.0)
    - 결과: **f1 score 상승 0.8392 → 0.8421**
- POS tagging 및 선택적 토큰 제거
    - konlpy 라이브러리의 Mecab 클래스를 이용해 text의 POS tagging 분석을 실행하였다. ‘신문 제목’이라는 text의 특성상 명사의 비중이 높고, 명사 토큰이 가지고 있는 정보량이 가장 많다고 가정하였다.
    - 조사, 형용사, 부사 등 비교적 정보량이 적은 토큰을 전부 제거하고 명사, 동사, 숫자, 한자, 영어만 남겨서 토큰 수 대비 정보량을 높였다.
    - 이때, subword 명사가 토큰화되는 과정에서 사이에 전부 공백이 들어가, ‘##’ 토큰이 소실되는 문제가 발생하였다. 이를 해결하기 위해 패턴 검사를 통해 원본 text에서 붙어 있던 토큰은 붙이고, 떨어져 있던 토큰은 떨어트려 정보의 훼손을 최소화하였다.
  
      ![Untitled (1)](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/1188cb2c-1c0d-4d3b-aa2d-2a198c4d903d)
      ![Untitled (2)](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/b4a58b11-9181-47f5-ae67-260324a2020b)
    
    - 결과: **f1 score 상승 0.8392 → 0.8431**
- 모든 특수문자 제거
    - 실험적으로 dot 특수문자가 아닌, 다른 특수문자도 전부 제거해 보았다. 그러나 ‘%’, ‘↑’ 등 유의미한 정보를 지닌 특수문자의 제거는 학습 성능의 저하로 이어졌다.
    - 결과: **f1 score 상승 0.8392 → 0.8405**
- [UNK] 토큰 제거
    - noise로 인해 ‘[UNK]’로 분석된 토큰을 전부 제거하였다.
    - 결과: **f1 score 상승 0.8392 → 0.8405**

하지만 모든 데이터 전처리 방법을 동시에 적용할 경우, train dataset의 기본 형태가 너무 달라져 test dataset과 차이가 커져 오히려 예측 성능이 하락하는 현상이 발생하였다.

따라서, 위의 방법들을 1~2개 선택적으로 적용하여 가장 성능이 좋은 방법을 선별하였다.

결과적으로, ‘POS tagging 및 선택적 토큰 제거’와 ‘[UNK] 토큰 제거’를 적용한 것이 가장 성능이 좋았으며 이를 최종 제출하였다.

 

### 1-4-4. 학습 데이터셋에 노이즈 추가

평가 데이터셋과 유사하게 학습 데이터셋을 만들기 위해 크롤링한 데이터셋에 G2P를 적용했다. 크롤링 데이터셋 중 20%에 노이즈를 추가했다.

- 결과: **f1 score 상승 0.8392 → 0.8410**

### 1-4-5. 모델 추론 결과 분석

![Untitled (3)](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/69b34507-c480-4f0f-bedb-63db5b55926c)

- 모델이 잘 틀리는 구간은 어디인지 확인함으로써 특정 label에 대한 labeling issue를 검사하거나 특정 topic에 대한 보완을 고민했다.

### 1-4-6. Prototyping

![SE-798aaa54-ef6b-46ef-8ebf-a5e477705395](https://github.com/boostcampaitech6/level2-nlp-datacentric-nlp-08/assets/76895949/b4c9d424-e833-4524-9ee2-77b5b91dce87)

- Streamlit을 사용해서 뉴스 제목을 입력하면 카테고리를 분류해주는 프로토타입 앱 개발

### 1-4-7. 추론 속도 향상

- baseline코드에서 데이터셋의 샘플들을 하나씩 inference하는 것을 배치 단위로 수행하도록 수정하여 output.csv를 생성하는 시간을 단축 (7분 → 30초)
