cd tools
python3 refine_text.py ../text/KCC150_Korean_sentences_UTF8.txt ../text/KCC150_refined_UTF8.txt 1
iconv -f utf8 -t cp949 ../text/KCC150_refined_UTF8.txt > ../text/KCC150_refined_CP949.txt
cd KLT2000-TestVersion/EXE
wine index2018.exe ../../../text/KCC150_refined_CP949.txt > ../../../text/KCC150_tokenized_CP949.txt
iconv -f cp949 -t utf8 ../../../text/KCC150_tokenized_CP949.txt > ../../../text/KCC150_tokenized_UTF8.txt
cd ../../../tools
python3 word_vector_train.py ../text/KCC150_tokenized_UTF8.txt