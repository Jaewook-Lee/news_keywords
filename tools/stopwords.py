import re


def get_stopwords() -> list[str]:
    """
    불용어를 가져오는 함수
    :return: 불용어들을 담은 list
    """
    stopwords_list = []
    with open("../text/stopwords.txt", "r") as f:
        words = f.readlines()
        stopwords_list.extend([word.strip() for word in words])

    return stopwords_list


def remove_special_chars(sentence: str) -> str:
    """
    정규 표현식을 사용해 영어, 특수 문자, 숫자를 제거하는 함수
    :param sentence: 전처리 대상인 단일 문장
    :return: 전처리 결과
    """
    pattern = re.compile(r"[a-zA-Z0-9!@#$%^&*(),.?\":{}|<> ]+")
    sentence = re.sub(pattern, " ", sentence).strip()
    return sentence
