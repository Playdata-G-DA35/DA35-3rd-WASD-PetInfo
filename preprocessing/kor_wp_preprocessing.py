from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE #Byte Pair Encoding알고리즘
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
import time
import os

class Kor_wp_preprocessing:
    def __init__(self, strpath, mname="wordpiece_tk", line=50_000, vsize=20_000):
        self.strpath = strpath
        self.mname = mname
        wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]")) #알고리즘 객체를 넣어서 tokenizer 생성 or BPE
        #pre tokenizer 설정.(1차 토큰화 작업단위) 공백단위로 쪼개놓고 
        wp_tokenizer.pre_tokenizer = Whitespace()
        #Trainer 생성 -> 어휘사전 어떻게 만들지 설정
        trainer = WordPieceTrainer(vocab_size=vsize)
        batch_size = line #몇 라인단위로 학습 시킬지.
        current_batch = [] #학습시킬 문자열들을 담을 리스트

        #json파일이 있으면 train
        self.file_path = f"saved_models/{self.mname}.json"
        os.makedirs("saved_models", exist_ok=True)
        if os.path.isfile(self.file_path):
            print("파일존재")
        else:
            with open(self.strpath, "rt", encoding="utf-8") as fr:
                s = time.time()
                # batch_size줄만큰 current_batch 리스트에 저장 한 뒤 학습.
                for line in fr:
                    #한줄씩 읽어보자
                    current_batch.append(line)
                    if len(current_batch) == batch_size:# 50000줄 batch size만큰 읽은 후 학습
                        #학습
                        wp_tokenizer.train_from_iterator(current_batch, trainer) #
                        #누적되므로 리스트 비우기
                        current_batch.clear()

                #current_batch에 있는 나머지 데이터 학습
                wp_tokenizer.train_from_iterator(current_batch, trainer)
                e = time.time()
                print(f"걸린시간: {e - s}초")
                wp_tokenizer.save(self.file_path)
    
    def get_wp_tokenizer(self, txt):
        saved_tokenizer = Tokenizer.from_file(self.file_path)
        return saved_tokenizer.encode(txt).tokens


if __name__ == "__main__" :
    print("main 시작!!")
    txt = "안녕하세요. 현재 사대, 교대 등 교원양성학교들의 예비교사들이 임용절벽에 매우 힘들어 하고 있는 줄로 압니다. 정부 부처에서는 영양사의 영양'교사'화, 폭발적인 영양'교사' 채용, 기간제 교사, 영전강, 스강의 무기계약직화가 그들의 임용 절벽과는 전혀 무관한 일이라고 주장하고 있지만 조금만 생각해보면 전혀 설득력 없는 말이라고 생각합니다. 학교 수가 같고, 학생 수가 동일한데 영양교사와 기간제 교사, 영전강 스강이 학교에 늘어나게 되면 당연히 정규 교원의 수는 줄어들게 되지 않겠습니까? 기간제 교사, 영전강, 스강의 무기계약직화, 정규직화 꼭 전면 백지화해주십시오. 백년대계인 국가의 교육에 달린 문제입니다. 단순히 대통령님의 일자리 공약, 81만개 일자리 창출 공약을 지키시고자 돌이킬 수 없는 실수는 하지 않으시길 바랍니다. 세계 어느 나라와 비교해도, 한국 교원의 수준과 질은 최고 수준입니다. 고등교육을 받고 어려운 국가 고시를 통과해야만 대한민국 공립 학교의 교단에 설 수 있고, 이러한 과정이 힘들기는 하지만 교원들이 교육자로서의 사명감과 자부심을 갖고 교육하게 되는 원동력이기도 합니다. 자격도 없는 비정규 인력들을 일자리 늘리기 명목 하에 학교로 들이게 되면, 그들이 무슨 낯으로 대한민국이 '공정한 사회' 라고 아이들에게 가르칠 수 있겠습니까? 그들이 가르치는 것을 학부모와 학생들이 납득할 수 있겠으며, 학생들은 공부를 열심히 해야하는 이유를 찾을 수나 있겠습니까? 열심히 안 해도 떼 쓰면 되는 세상이라고 생각하지 않겠습니까? 영양사의 영양교사화도 재고해주십시오. 영양사분들 정말 너무나 고마운 분들입니다. 학생들의 건강과 영양? 당연히 성장기에 있는 아이들에게 필수적이고 중요한 문제입니다. 하지만 이들이 왜 교사입니까. 유래를 찾아 볼 수 없는 영양사의 '교사'화. 정말 대통령님이 생각하신 아이디어라고 믿기 싫을 정도로 납득하기 어렵습니다. 중등은 실과교과 교사가 존재하지요? 초등 역시 임용 시험에 실과가 포함돼 있으며 학교 현장에서도 정규 교원이 직접 실과 과목을 학생들에게 가르칩니다. 영양'교사', 아니 영양사가 학생들에게 실과를 가르치지 않습니다. 아니 그 어떤 것도 가르치지 않습니다. 올해 대통령님 취임 후에 초등, 중등 임용 티오가 초전박살 나는 동안 영양'교사' 티오는 폭발적으로 확대된 줄로 압니다. 학생들의 교육을 위해 정말 교원의 수를 줄이고, 영양 교사의 수를 늘리는 것이 올바른 해답인지 묻고 싶습니다. 마지막으로 교원 당 학생 수. 이 통계도 제대로 내주시기 바랍니다. 다른 나라들은 '정규 교원', 즉 담임이나 교과 교사들로만 통계를 내는데(너무나 당연한 것이지요) 왜 한국은 보건, 영양, 기간제, 영전강, 스강 까지 다 포함해서 교원 수 통계를 내는건가요? 이런 통계의 장난을 통해 OECD 평균 교원 당 학생 수와 거의 비슷한 수준에 이르렀다고 주장하시는건가요? 학교는 교육의 장이고 학생들의 공간이지, 인력 센터가 아닙니다. 부탁드립니다. 부디 넓은 안목으로 멀리 내다봐주시길 간곡히 부탁드립니다."
    
    wptoken = Kor_wp_preprocessing("data/petitions_corpus.txt")
    tok = wptoken.get_wp_tokenizer(txt)
    print(tok)