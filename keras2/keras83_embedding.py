from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재미있어요", '참 최고에요', '참 잘 만든 영화에요', "추천하고 싶은 영화입니다.", "한 번 더 보고 싶네요", "글세요", "별로에요", "생각보다 지루해요", "연기가 어색해요", "재미없어요", "너무 재미없다", "참 재밋네요", "규현이가 잘생김 뻔 했어요"]


# 긍정 1, 부정 0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x= token.texts_to_sequences(docs)

print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
# 0으로 길이를 맞춰주는 전처리 방법 

pad_x = pad_sequences(x, padding="pre", maxlen=5) # post
# 데이터를 컷하는 방법 앞 뒤 선택 가능


print(pad_x)
print(pad_x.shape) #(13,4)

print(np.unique(pad_x))
print(len(np.unique(pad_x))) # 28 / 0부터 27까지인데 11이 maxlen=4로 인해 잘렸다.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()    

#model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# input_dim 단어의 갯수, ouput_dim 노드의 수 (아무 숫자 가능), input_length 자리 수

model.add(Embedding(28,11))
# 이 둘의 파라미터는 308로 같다. 이는 총 단어의 수 * 내가 지정한 아웃풋 딤 = 28 * 11 = 308이다.


#model.add(Flatten())
#model.add(Dense(1))

model.summary()


