def tokenize_text(text):
    """
    전체 문장들을 토큰화해 반환하는 함수
    문장별로 단어 리스트(의미를 파악하는데 중요한 단어들)를 2차원 배열 형태로 반환
    구두점/특수문자, 숫자, 불용어(stop words)들은 모두 제거한다.
    [매개변수]
        text: string - 변환하려는 전체문장
    [반환값]
        2차원 리스트. 1차원: 문장 리스트, 2차원: 문장내 토큰.
    """
    #1. 받은 문장을 모두 소문자로 변환.
    text = text.lower()
    #2. 문장단위 토큰화
    sent_tokens = nltk.sent_tokenize(text)
    #3. 클린징 작업 - 불용어, 특수문자, 숫자 등등을 제거
    ## 불용어사전 loading
    stop_words = stopwords.words("english")
    stop_words.extend(['', '', '']) # 필요하다면 불용어 단어를 추가할 수있다.
    result_list = [] # 최종 결과를 저장할 리스트
    for sent in sent_tokens:  # 한문장씩 처리
        result = nltk.regexp_tokenize(sent, r"[A-Za-z]+")
        # 불용어제거
        result = [word for word in result if word not in stop_words]
        result_list.append(result)
    
    return result_list