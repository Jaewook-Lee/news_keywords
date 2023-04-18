"""
텍스트 파일에 대해 TF-IDF를 적용한 결과를 반환하는 함수를 담은 모듈\n
- get_top_n_words: TF-IDF 결과 상위 N개를 반환하는 함수
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame


def get_top_n_words(filename: str, entries: int) -> list[str]:
    """
    텍스트 파일에 대한 TF-IDF 결과들 중 상위 N개를 반환하는 함수
    :param filename: TF-IDF 계산 대상인 텍스트 파일 이름
    :param entries: 반환 받을 토큰의 수
    :return: entries 값 만큼의 상위 TF-IDF 토큰들을 담은 list
    """
    docs = []
    with open(filename, "r") as f:
        text = f.readlines()
        for sentence in text:
            docs.append(sentence.strip())

    tf_idf_vector = TfidfVectorizer().fit(docs)
    result = tf_idf_vector.transform(docs).toarray()
    vocabs = tf_idf_vector.vocabulary_

    dataframe_col = []
    for word, idx in sorted(vocabs.items(), key=lambda x: x[1]):
        dataframe_col.append(word)

    tf_idf_dataframe = DataFrame(result, columns=dataframe_col)
    try:
        top_n_series = tf_idf_dataframe.sum().sort_values(ascending=False)[:entries]
        return list(top_n_series.index)
    except IndexError:
        print("요청하신 단어의 수가 너무 많습니다. 양을 줄여서 다시 요청하세요.")
        exit()


if __name__ == "__main__":
    get_top_n_words("../text/news.txt", 30)
