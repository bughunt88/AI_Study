from tensorflow.keras.datasets import reuters, imdb
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

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000) 

# [실습/ 과제] Embadding으로 모델 만들 것


# # print(x_train[0], type(x_train[0]))
# # print(y_train[0])
# print(len(x_train[0]), len(x_train[11])) 
# print('====================================')
# print(x_train.shape, x_test.shape) 
# print(y_train.shape, y_test.shape) 

# print('뉴스기사 최대길이 : ', max(len(l) for l in x_train)) #  최대길이 :  2494
# print('뉴스기사 평균길이 : ', sum(map(len, x_train))/ len(x_train)) #  평균길이 :  238.71364

# # plt.hist([len(s) for s in x_train], bins = 40)
# # plt.show() # x가 데이터 길이

# # y분포
# unique_elements, counts_elements = np.unique(y_train, return_counts=True) # 해석 써놓기
# print('y분포 :', dict(zip(unique_elements, counts_elements)))
# print('===============================================')

# #plt.hist([len(s) for s in y_train], bins = 46)
# #plt.show() # y가 데이터 길이

# # x의 단어들 분포
# word_to_index = reuters.get_word_index()
# # print(word_to_index)
# # print(type(word_to_index)) # 단어 유형 --> word_size = input_dim
# # print('----------------------------------------')

# # 키와 밸류를 교체
# index_to_word = {} # 딕셔너리 하나 생성
# for key, value in word_to_index.items():
#     index_to_word[value] = key

# # 키 밸류 교환 후 
# # print(index_to_word) # 'mdbl': 10996,  --> 10996: 'mdbl'
# # print(index_to_word[1]) # the 가장 많이 썼다는 뜻
# print(len(index_to_word)) # 30979
# print(index_to_word[30979])

# # x_train[0]
# print(x_train[0])
# print(' '.join([index_to_word[index] for index in x_train[0]]))



# ================= 전처리 ==============================
x_train = pad_sequences(x_train, padding='pre', maxlen = 400)
x_test = pad_sequences(x_test, padding='pre', maxlen = 400)


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape) 

# (25000, 400) (25000, 400)
# (25000,) (25000,)

# ================= 모델링 ===========================
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim = 128, input_length=400))
model.add(LSTM(128, activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

es = EarlyStopping(monitor= 'val_loss', mode= 'min', verbose = 1, patience= 10)

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size = 32, validation_split=0.2, callbacks =[es])

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('정확도 : ', acc)