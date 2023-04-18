"""
네이버 뉴스 기사를 크롤링 해서 파일로 저장하는 함수를 담은 모듈\n
- save_news_content: 뉴스 기사 크롤링, 파일로 저장하는 함수
"""
from bs4 import BeautifulSoup
import requests
import sys
from tqdm import tqdm


def save_news_content(url: str, dest: str):
    """
    네이버 뉴스 기사 본문을 크롤링해서 파일로 저장하는 함수\n
    뉴스 기사 주소는 다음의 형식을 따르는 주소만 크롤링이 가능\n
    https://n.news.naver.com/mnews/article/<이하 경로>
    :param url: 네이버 뉴스 기사 주소
    :param dest: 작성될 파일 경로
    :return: 없음
    """
    response = requests.get(url, headers={"User-Agent": "Mozila/5.0"})
    html = BeautifulSoup(response.text, "html.parser")
    contents = html.find("div", {"id": "dic_area"}).text.split(".")
    refined = [content.strip() for content in contents]
    with open(dest, "w") as f:
        for sentence in tqdm(refined):
            f.write(sentence + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("$ news_crawling.py <DESTINATION FILE PATH>")
        exit()

    news_url = input("Enter URL to crawling > ")
    destination = sys.argv[1]
    save_news_content(news_url, destination)
