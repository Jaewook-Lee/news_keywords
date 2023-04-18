"""
Word2Vec 모델을 훈련시키는 함수를 담은 모듈\n
- tokens_train: Word2Vec 모델 훈련 함수
"""
from gensim.models import word2vec
import sys


def tokens_train(filename):
    """
    Word2Vec 모델 훈련 함수
    :param filename: 훈련에 사용할 대용량 텍스트 파일 이름
    :return: 없음
    """

    # 훈련에 필요한 토큰들 저장
    tokens = []
    with open(filename, "r") as f:
        print(f"Training word embedding vectors for <{filename}>.")
        text = f.readlines()
        for sent in text:
            tokens.append(sent.split())

    if not tokens:
        print("No token for result.")
        print("Are you sure that you tokenized well?")
        exit()

    wv_model = word2vec.Word2Vec(sentences=tokens, vector_size=300, window=5, min_count=2, epochs=30, workers=4)
    model_file = 'word2vec-' + filename[8: len(filename) - 4] + '.model'
    wv_model.save("../model/" + model_file)
    print(f"--> Model file <{model_file}> was created!\n")
    return wv_model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("$ word_vector_train.py <YOUR FILE NAME>")
        exit()
    model = tokens_train(sys.argv[1])

    # Model 테스트
    print("\'남한\' 벡터와 \'북한\' 벡터 간의 유사도:", model.wv.similarity("남한", "북한"))
    print(model.wv.most_similar("연예인"))
