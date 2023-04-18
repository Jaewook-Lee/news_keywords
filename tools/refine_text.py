"""
텍스트에 전처리 과정을 적용하기 위해 필요한 함수들을 담은 모듈\n
- make_unique: 중복되는 문장들 제거하는 함수\n
- get_random_indices: 랜덤하게 고른 인덱스들을 반환하는 함수\n
- refine_text: 불용어 등의 필요 없는 값들을 제거하는 함수\n
"""
import sys
import random
from tqdm import tqdm
from stopwords import get_stopwords, remove_special_chars


def make_unique(filename: str):
    """
    텍스트 문장들 중 중복되는 문장을 지우는 함수
    :param filename: 파일 이름
    :return: 없음
    """
    sentences = []
    with open(filename, "r") as f:
        text = f.readlines()
        for sentence in tqdm(text):
            sentences.append(sentence.strip())

    # 중복되는 문장 제거를 위해 set으로 전환 후 후에 다시 list로 변환
    return list(set(sentences))


def get_random_indices(total_range: int, max_entry: int) -> set:
    """
    전체 범위 안에서 N개의 랜덤한 인덱스 값을 반환하는 함수
    :param total_range: 인덱스의 전체 범위
    :param max_entry: 랜덤하게 뽑을 인덱스의 수
    :return: 랜덤하게 뽑힌 인덱스를 담은 집합
    """
    indices = set()
    while len(indices) < max_entry:
        indices.add(random.randint(0, total_range))

    return indices


def refine_text(text: list[list[str]], stop_words: list[str]) -> list[list[str]]:
    """
    텍스트에서 특수 문자, 영어, 숫자, 불용어 제거한 결과를 반환하는 함수
    :param text: 원본 문장들을 담고 있는 2차원 list
    :param stop_words: 불용어를 담고 있는 list
    :return: 전처리가 완료된 문장들을 담은 2차원 list
    """
    tokens = []
    for sentence in tqdm(text):
        if len(sentence) == 0:
            continue
        sentence = remove_special_chars(sentence)
        tokens.append([token for token in sentence.split() if token not in stop_words])

    new_text = ""
    for token_line in tokens:
        refined_sentence = " ".join(token_line)
        new_text += (refined_sentence + "\n")

    return tokens


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("$ python3 refine_text.py <YOUR FILE PATH> <DESTINATION PATH> <SAMPLING FLAG>")
        exit()

    print("중복되는 문장들을 지우는 중입니다...")
    input_file_name = sys.argv[1]
    unique_sentences = make_unique(input_file_name)

    # 용량 문제로 인해 일부 문장만 샘플링
    # 이 과정은 워드 임베딩 모델을 학습하기 위한 말뭉치에만 해당
    if sys.argv[3] == "1":
        print("일부 문장들만 샘플링 하는 중입니다...")
        samples = []
        origin_length = len(unique_sentences)
        index_set = get_random_indices(origin_length, int(origin_length * 0.5))    # 50%만큼만 사용
        for i, sentence in tqdm(enumerate(unique_sentences)):
            if i in index_set:
                samples.append(sentence)

        # 샘플링 결과가 0개라면 입력으로 준 파일에 문제가 있을 것이라 추정
        # 프로그램 종료
        if not samples:
            print("Something wrong in sampling.")
            print("Are you sure that your file(%s) is not empty?" % input_file_name)
            exit()
    else:
        samples = unique_sentences

    print("불용어 및 영문자와 특수 문자를 제거 중입니다...")
    stop_words_list = get_stopwords()
    refined_sentences = refine_text(samples, stop_words_list)
    with open(sys.argv[2], "w") as f:
        for sentence in refined_sentences:
            refined_sentence = " ".join(sentence)
            f.write(refined_sentence + "\n")

    print("정제된 텍스트를 저장했습니다.\n")
