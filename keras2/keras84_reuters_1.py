from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words = 1000, test_split = 0.2
)

print(x_train[0], type(x_train[0])) # [1, 2, 2, 8, 43, 10, 447, ... ]  <class 'list'>
print(y_train[0]) # 3
print(len(x_train[0]), len(x_train[11])) # 87 59
print('================================================')
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print(np.min(y_train),np.max(y_train)) # 0 45 >> 46개의 뉴스 카테고리로 나눠준다

print("뉴스기사 최대길이 : ", max(len(l) for l in x_train)) # for 문 돌리면서 제일 긴 문장길이 추출
print("뉴스기사 평균길이 : ", sum(map(len, x_train)) / len(x_train)) 
# 뉴스기사 최대길이 :  2376
# 뉴스기사 평균길이 :  145.5398574927633

# plt.hist([len(s) for s in x_train], bins = 50) # 분포 확인!
# plt.show()

# y분포
unique_elements, counts_elements = np.unique(y_train, return_counts = True)
print('y분포 : ', dict(zip(unique_elements, counts_elements)))
print('================================================')
# plt.hist(y_train, bins = 46) # 분포 확인!
# plt.show()

# x의 단어들 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index)) # <class 'dict'>
print('================================================')

# 키와 밸류를 교체!
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
# 키 밸류 교환 후
print(index_to_word)
print(index_to_word[15])
print(index_to_word[30979])
print(len(index_to_word)) # 30979

# x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))

# y 카테고리 갯수 출력
category = np.max(y_train) + 1
print('y 카테고리 개수 : ', category) # y 카테고리 개수 : 46

# y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 100, padding= 'pre')
x_test = pad_sequences(x_test, maxlen = 100, padding= 'pre')
print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, Flatten

model = Sequential()
# model.add(Embedding(input_dim = 1000, output_dim = 64, input_length = 100))
model.add(Embedding(10000, 64))
model.add(LSTM(32))
model.add(Dense(46, activation = 'softmax'))

model.summary()

# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

model.fit(x_train, y_train, epochs = 10, batch_size = 32)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])