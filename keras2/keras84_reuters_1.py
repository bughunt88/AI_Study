# 단어 input_dim = 10000, maxlen 자르는 방법, 임베딩사용해서 모델링
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 5000, test_split=0.2
) # num_words = 10000, 10000번째 안에 있는 것을 가져온다

# ================= 전처리 ==============================
x_train = pad_sequences(x_train, padding='pre', maxlen = 500)
x_test = pad_sequences(x_test, padding='pre', maxlen = 500)

y_train = to_categorical(y_train)        # y도 원핫인코딩 꼭 하기
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)
print(y_train.shape, y_test.shape) # (8982, 100) (2246, 100)

# ================= 모델링 ===========================
model = Sequential()
model.add(Embedding(input_dim = 5000, output_dim = 128, input_length=500))
model.add(LSTM(128, activation='tanh'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(32,activation='tanh'))
model.add(Dense(28,activation='tanh'))
model.add(Dense(46,activation='softmax'))
model.summary()

es = EarlyStopping(monitor= 'val_loss', mode= 'min', verbose = 1, patience= 10)

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size = 32, validation_split=0.2, callbacks =[es])

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('정확도 : ', acc)


