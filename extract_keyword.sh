cd tools
python3 news_crawling.py ../text/news.txt
python3 refine_text.py ../text/news.txt ../text/news_refined_UTF8.txt 0
iconv -f utf8 -t cp949 ../text/news_refined_UTF8.txt > ../text/news_refined_CP949.txt
cd KLT2000-TestVersion/EXE
wine index2018.exe ../../../text/news_refined_CP949.txt > ../../../text/news_tokenized_CP949.txt
iconv -f cp949 -t utf8 ../../../text/news_tokenized_CP949.txt > ../../../text/news_tokenized_UTF8.txt
cd ../../../
python3 extract_keyword_from_news.py word2vec-KCC150_tokenized_UTF8.model text/news_tokenized_UTF8.txt