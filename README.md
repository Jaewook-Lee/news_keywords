# TF-IDF와 워드 임베딩을 사용한 뉴스 키워드 추출
본 프로젝트는 TF-IDF와 워드 임베딩을 같이 사용해 네이버 뉴스 기사에서 키워드를 추출하는 프로젝트입니다.  
하지만 성능은 좋지 않기 때문에 이 코드를 성능의 목적으로 사용하는 것을 추천하지 않습니다.
## 코드 실행 전 준비 사항
코드를 실행하기 전에 두 가지 텍스트 파일과 형태소 분석기를 준비해야 합니다.
### Word2Vec 모델 학습을 위한 텍스트 파일
word2vec 모델을 학습시킬 대용량 텍스트 파일을 준비해야 합니다.  
준비한 텍스트 파일은 아래 프로젝트 구조의 text 디렉터리 안에 저장하셔야 합니다.
### 불용어 텍스트 파일
전처리 과정에서 사용할 불용어들을 담은 텍스트 파일을 준비해야 합니다.  
프로젝트에서 사용한 불용어는 [사이트](https://www.ranks.nl/stopwords/korean)를 통해 얻을 수 있습니다.  
이 파일 또한 text 디렉터리 안에 저장해야 합니다.
### 형태소 분석기(KLT2000)
형태소 분석기 __KLT2000-TestVersion__은 국민대학교 소프트웨어학부 강승식 교수님이 제공한 형태소 분석기입니다.  
형태소 분석기는 [네이버 카페 사이트](https://cafe.naver.com/nlpkang)에서 얻을 수 있습니다.  
얻은 형태소 분석기는 tools 디렉터리 안에 저장해야 합니다.
## 코드 실행 방법
코드는 shell script 파일을 통해 실행됩니다.  
만약 shell script 파일을 실행할 수 없는 경우, 파일 내부에 작성된 순서에 맞게 터미널에 입력하면서 실습할 수 있습니다.  
이 프로젝트를 실습하기 위해서는 모델 학습 과정을 먼저 거쳐야 합니다.
모델의 학습은 __model_training.sh__ 파일을 실행해서 진행할 수 있습니다.  
모델의 학습 파라미터를 바꾸고 싶다면 __word_vector_train.py__ 코드에서 변경할 수 있습니다.  
학습이 끝나면 __extract_keyword.sh__ 파일을 실행해서 키워드를 확인할 수 있습니다.  
출력은 _My result_ 와 _TF-IDF result_ 두 가지 있습니다.  
'My result'는 TF-IDF와 워드 임베딩을 모두 적용해서 얻은 키워드를 표현하고,  
'TF-IDF result'는 TF-IDF만 적용할 때 얻은 키워드를 표현한 것입니다.
## 작업 환경
- 운영체제: Ubuntu 22.04.2 LTS
- 프로그래밍 언어: Python3
- 작업 IDE: Pycharm Community 2023
## 사전 설치 패키지
- beautifulsoup4: 4.12.2
- gensim: 4.3.1
- matplotlib: 3.7.1
- numpy: 1.24.2
- pandas: 2.0.0
- scikit-learn: 1.2.2
- tqdm: 4.65.0
## 프로젝트 구조
```
-- model
|
-- text
|
-- tools
|    |
|    -- KLT2000-TestVersion
|    -- draw_vector.py
|    -- news_crawling.py
|    -- refine_text.py
|    -- stopwords.py
|    -- tf_idf.py
|    -- word_vector_train.py
-- extract_keyword_from_news.py
|
-- extract_keyword.sh
|
-- model_training.sh
|
-- README.md
```