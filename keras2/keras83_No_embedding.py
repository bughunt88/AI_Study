from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다',
        '한 번 더 보고 싶네요', '글쎄요', '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘 생기긴 했어요']

# 긍정1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)

print("#####################")

from tensorflow.keras.preprocessing.sequence import pad_sequences
# 0으로 길이를 맞춰주는 전처리 방법 

pad_x = pad_sequences(x, padding="pre", maxlen=5) # post
# 데이터를 컷하는 방법 앞 뒤 선택 가능
print(pad_x)
print(pad_x.shape) # (13, 4)

print(np.unique(pad_x)) # 하나씩만 추출해서 리스트화 - 11이 없다!
print(len(np.unique(pad_x))) # 27


pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()


# LSTM
# model.add(LSTM(10, activation='relu', input_shape=(5,1))) 

# Conv1D
model.add(Conv1D(filters=64, kernel_size=(2), padding='same', input_shape=(5,1), activation='relu'))



model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(pad_x, labels, epochs = 30)

acc = model.evaluate(pad_x, labels)[1]
print(acc)

