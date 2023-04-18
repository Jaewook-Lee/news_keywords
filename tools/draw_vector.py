"""
Dataframe을 matplotlib 패키지를 통해 시각화하는 함수들을 담은 모듈\n
- get_pca_df: PCA 방법이 적용된 Dataframe을 반환하는 함수\n
- scatter_pca_vector: PCA 방법이 적용된 Dataframe을 scatter하는 함수\n
- scatter_clustered: Cluster 레이블을 가진 Dataframe을 scatter 하는 함수
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_pca_df(dataframe: pd.DataFrame) -> tuple:
    """
    PCA 방법을 통해 고차원의 데이터들을 저차원으로 축소시키는 함수
    :param dataframe: PCA를 적용시킬 Dataframe 객체
    :return: PCA를 적용한 Dataframe, PCA 컴포넌트의 column 이름들
    """
    scaled_dataframe = StandardScaler().fit_transform(dataframe)    # 정규화
    pca = PCA(n_components=2)    # 2차원으로 축소
    columns = ["PCA Components 0", "PCA Components 1"]
    vector_pca = pd.DataFrame(pca.fit_transform(scaled_dataframe), columns=columns)
    return vector_pca, columns


def scatter_pca_vector(dataframe: pd.DataFrame):
    """
    PCA를 적용시킨 dataframe을 scatter plot으로 표현하는 함수
    :param dataframe: PCA가 적용된 Dataframe
    :return: 없음
    """
    vector_pca, cols = get_pca_df(dataframe)
    plt.scatter(vector_pca[cols[0]], vector_pca[cols[1]])
    plt.title("Word Vectors with PCA")
    plt.show()


def scatter_clustered(dataframe: pd.DataFrame, by: str):
    """
    클러스터링 알고리즘이 적용된 Dataframe을 scatter plot으로 표현하는 함수
    :param dataframe: 클러스터 레이블을 가지고 있는 Dataframe
    :param by: 클러스터 기준 column 이름
    :return: 없음
    """
    marker_colors = ["red", "orange", "green", "blue", "purple", "black", "pink", "aquamarine", "magenta", "slategray"]
    markers = [".", "^", "*", "v", "<", ">", "p", "+", "x", "H"]
    unique_labels = np.unique(dataframe[by].values)
    for label in unique_labels:
        cluster = dataframe[dataframe[by] == label]
        if label == -1:
            # Noise 데이터들을 scatter
            plt.scatter(cluster["PCA Components 0"], cluster["PCA Components 1"], marker='x', c="black", label="Noise")
        else:
            plt.scatter(cluster["PCA Components 0"], cluster["PCA Components 1"], marker=markers[label],
                        c=marker_colors[label], label="Cluster %d" % label)
    plt.legend(loc="upper right")
    plt.show()
