"""
뉴스 기사에서 키워드를 추출하는 프로세스를 작성한 코드\n
이에 필요한 2가지 함수도 존재\n
- cosine_similarity: 두 벡터 간의 유사도를 계산하는 함수\n
- select_keyword: 클러스터 안에서 대표 키워드를 선택하는 함수\n
"""
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import sys

from tools.tf_idf import get_top_n_words
from tools.draw_vector import get_pca_df, scatter_pca_vector, scatter_clustered


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 벡터 간의 코사인 유사도를 계산하는 함수
    sklearn에 이미 계산하는 함수가 있지만 벡터 차원 문제로 인해 직접 식을 작성
    :param a: 계산 대상 벡터
    :param b: 계산 대상 벡터
    :return: 두 벡터 간의 코사인 유사도
    """
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm


def select_keyword(_model: Word2Vec, cluster_words: list['str'], mean_vector: pd.DataFrame) -> str:
    """
    클러스터 안에서 대표 키워드를 선정하는 함수
    :param _model: Word2Vec 모델
    :param cluster_words: 하나의 클러스터에 담긴 토큰들을 담은 list
    :param mean_vector: 한 클러스터의 평균 벡터를 담은 Dataframe
    :return: 클러스터의 대표 키워드
    """
    max_similarity = -1
    keyword = None
    mean_vector = mean_vector.to_numpy()

    # 평균 벡터와 가장 높은 유사도를 가진 키워드 탐색
    for i in range(len(cluster_words)):
        word_vector = _model.wv.get_vector(cluster_words[i])
        cos_similarity = cosine_similarity(mean_vector, word_vector)
        if cos_similarity > max_similarity:
            max_similarity = cos_similarity
            keyword = cluster_words[i]

    return keyword


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("$ extract_keyword_from_news.py word2vec.model tokenized_test.txt")
        exit()

    # 모델 로딩
    print("Loading Korean word embedding vectors for 'KMA tokenized text file'.\n")
    model_name = sys.argv[1]    # Word2Vec model -- 'word2vec-kowiki.model'
    model = Word2Vec.load("./model/" + model_name)

    # 뉴스 기사 로딩
    print("Loading Korean news tokens for test\n")
    file_name = sys.argv[2]
    top_30_words = get_top_n_words(file_name, 30)

    # 모델에 있는 벡터들만 선택
    mapped_vectors = {}
    for i in range(len(top_30_words)):
        word = top_30_words[i]
        try:
            mapped_vectors[word] = model.wv.get_vector(word)
        except KeyError:
            continue
    vector_dataframe = pd.DataFrame(mapped_vectors.values(), index=mapped_vectors.keys())

    # DBSCAN 클러스터링 적용
    dbscan = DBSCAN(eps=25, min_samples=4)
    dbscan_labels = dbscan.fit_predict(vector_dataframe)

    # 벡터들 시각화(PCA 적용)
    # pca_vector, _ = get_pca_df(vector_dataframe.iloc[:, :])
    # scatter_pca_vector(pca_vector)

    # 클러스터 결과 시각화(PCA 적용)
    # pca_vector["cluster index"] = dbscan_labels
    # scatter_clustered(pca_vector, by="cluster index")

    # Noise 제외한 cluster 저장
    # 단, cluster가 1개면 모든 토큰을 저장
    num_cluster = len(np.unique(dbscan_labels))
    clusters = [[] for _ in range(num_cluster)]
    vector_dataframe["cluster index"] = dbscan_labels
    for index, row in vector_dataframe.iterrows():
        if num_cluster != 1 and row["cluster index"] == -1:    # Noise면 cluster index값이 -1
            continue
        else:
            clusters[0 if num_cluster == 1 else int(row["cluster index"])].append(index)

    # 클러스터 출력
    for i in range(len(clusters)):
        print("Cluster %d:" % i, clusters[i] if len(clusters[i]) else "Noise")

    # 클러스터 별 키워드 선정
    mean_vectors = vector_dataframe.groupby("cluster index").mean()
    keywords = []
    for i in range(num_cluster):
        if len(clusters[i]) == 0:
            continue
        else:
            keywords.append(select_keyword(model, clusters[i], mean_vectors.iloc[i]))

    print("My result:", keywords)
    print("TF-IDF result:", top_n_words := get_top_n_words(file_name, 5))
    print("공통 키워드:", result := set(keywords) & set(top_n_words))
