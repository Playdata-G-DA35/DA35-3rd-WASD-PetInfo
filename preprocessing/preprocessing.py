import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag

from nltk import Text
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class Text_preprocess:

    def __init__(self, path, column):
        self.path = path
        self.column = column

    def get_data(self):
        df = pd.read_csv(self.path)
        breed_info = df[self.column].copy()
        tok_result = []
        for word in breed_info:
            tok_result += self.tokenize_text(str(word))
        # 중첩되어있는 2차원 리스트를 1차원으로 바꿔야함
        dim_one_words = []
        for lst in tok_result:
            dim_one_words += lst
        return dim_one_words
    
    def pet_to_text(self, dim_one_words):
        return Text(dim_one_words)
    
    def pet_word_freq(self, dim_one_words):
        # pet_text = Text(dim_one_words)
        pd_dim = pd.DataFrame(dim_one_words)
        as_sort = pd_dim.groupby([0]).size().sort_values(ascending=False).copy()
        copy_as = as_sort.iloc[:20].copy()
        
        plt.figure(figsize=(13,4))
        plt.xlabel("freq")
        plt.ylabel("pet")
        plt.title("Line Plot: Series Index vs. Values")
        plt.plot(copy_as.index, copy_as.values, marker="o", linestyle="-", color="b")
        plt.savefig('pet_word_freq.png')
        plt.show()



    def tokenize_text(self, text):
        """text 토큰화 처리 함수"""
        #1.소문자 변환
        text = text.lower()
        #2. 문장단위로 토큰화
        sent_tokens = nltk.sent_tokenize(text) #[문장1, 문장2, 문장3,..]
        ###토큰화 + cleaning작업
        #3. stopword들 loading
        stop_words = stopwords.words("english") # 필요하다면 더 추가, 제거할 수도 있다.
        stop_words.extend(["although", "unless", "may", ",", "'s", "'", ".", "\n"])
        #4. lemmatizer객체 생성
        lemm = WordNetLemmatizer()
        #최종 결과를 담을 리스트
        result_tokens = []
        #문장별로 처리
        for sent in sent_tokens:
            #regexp 토큰화
            word_tokens = nltk.regexp_tokenize(sent, r"[a-zA-Z]+") #[단어1, 단어2, 단어3, ..]
            #불용어 제거
            word_tokens = [word for word in word_tokens if word not in stop_words]
            #원형 복원
            ##품사부탁
            word_tokens = pos_tag(word_tokens) #[(단어1, 품사), (단어2, 품사), (단어3, 품사),..]
            ##lemmatizer에서 사용하는 품사 string으로 변환.: "NN" -> "n"
            word_tokens = [ (word, self.get_wordnet_pos(pos)) \
                           for word, pos in word_tokens \
                           if self.get_wordnet_pos(pos) is not None]
            ##원형복원
            word_tokens = [lemm.lemmatize(word, pos=pos) for word, pos in word_tokens]
            result_tokens.append(word_tokens)
        return result_tokens

    # Pos-tag 에서 반환한 품사표기(펜 트리뱅크 태그세트)을 WordNetLemmatizer의 품사표기로 변환
    def get_wordnet_pos(self, pos_tag):
        """
        펜 트리뱅크 품사표기를 WordNetLemmatizer에서 사용하는 품사표기로 변환
        형용사/동사/명사/부사 표기 변환
        """
        if pos_tag.startswith("J"):
            return wordnet.ADJ
        elif pos_tag.startswith("V"):
            return wordnet.VERB
        elif pos_tag.startswith("N"):
            return wordnet.NOUN
        elif pos_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None
        
    def pet_dispersion(self, pet_list):
        # self.pet_word_freq(self.get_data())
        pet_text = self.pet_to_text(self.get_data())
        # ##특정 단어가 문서의 어느부분들에 나오는지 시각화
        # plt.figure(figsize=(10,10))
        pet_text.dispersion_plot( ##내부적으로 figsize를 정할수 없었다
            pet_list
        )
        plt.savefig('pet_dispersion.png')
        plt.show()

    def pet_word_cloud(self):
        fd = self.pet_to_text(self.get_data()).vocab()
        #WordCloud 객체 생성 -> 어떻게 그릴지 설정
        wc = WordCloud(
            max_words = 50, # 최대 몇개단어를 사용할지설정. 빈도수 많은 순서대로
            prefer_horizontal=0.7, #가로쓰기 비율(기본:0.9)
            width=500, #가로크기:픽셀
            height=500, #세로크기:픽셀
            relative_scaling=0.5, #빈도수가 증가할 때 글씨 크기를 얼마나 크게 할지 비율
                                #(단계적 크기: 1이면 두배)
            min_font_size=30,
            max_font_size=100,
            background_color="white"
        )
        word_cloud = wc.generate_from_frequencies(fd)
        word_cloud.to_file("pet_info_wordcloud.png")
        plt.imshow(word_cloud)
        plt.xticks([])
        plt.yticks([])
        plt.show()

if __name__ == "__main__" :
    print("main 시작!!")

    tp = Text_preprocess("../data_pet_csv/happy_puppy_info.csv", "Breed_info")
    b_info_list = tp.get_data()
    # print(b_info_list)
    # tp.pet_word_freq(b_info_list) 
    # tp.pet_dispersion(["dog", "coat", "breed", "inch", "terrier"])
    # tp.pet_word_cloud()
