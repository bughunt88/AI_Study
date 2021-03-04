# (25000,) (25000,)
# 이진 분류
# [실습 / 과제] Embedding 으로 모델 만들 것!
# print("뉴스기사 최대길이 : ", max(len(l) for l in x_train)) # 2494
# print("뉴스기사 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 238.71

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

#1. 데이터
maxlen = 240
vocab = 2000

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = vocab
)

x_train = pad_sequences(x_train, maxlen = maxlen)
x_test = pad_sequences(x_test, maxlen = maxlen)

#2. 모델
model = Sequential()
model.add(Embedding(vocab, 100, input_length = maxlen))
model.add(LSTM(128))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일 훈련
es = EarlyStopping(patience = 10)
lr = ReduceLROnPlateau(factor = 0.25, patience = 5, verbose = 1)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, validation_split = 0.2, epochs = 1000, callbacks = [es, lr])

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('acc : ', acc)

# acc :  0.8626400232315063