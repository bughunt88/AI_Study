# keras21_cancer1.py 를 다중분류로 코딩하시오.


# 이중 분류 -> 다중 분류 


# 실습1. acc 0.985이상 올릴 것
# 실습2. predict 출력해볼 것

'''
y[-5:-1] = ? 예상 0 아니면 1
y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[[-5:-1]])
'''



import numpy as np
from sklearn.datasets import load_breast_cancer


# 1. 데이터 

datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

# 데이터를 받으면 컬럼이 몇인지 부터 확인한다

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다

# print(x.shape) #(569,30)
# print(y.shape) #(569,)

# print(x[:5])
# print(y)

# 전처리 알아서 할 것 / minmax, train_test_split


from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical
# 위 아래 코드는 동일함 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


# 2. 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(169, input_shape=(30,), activation='relu'))
model.add(Dense(169, activation='relu'))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(2, activation='softmax'))
# sigmoid는 0~1 사이로 한정을 하는 코드
# 히든이 없는 모델도 가능하다 


# 3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# metrics에 acc도 쓸 수 있고 accuracy 로 쓸 쑤 있다


from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다
# loss 값이 10번 최하로 떨어지면 그 지점에서 멈춘다 
# patience에 지정한 수 만큼 최저점이 지나가서 멈춘다 (보강을 해야한다)

model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


# loss의 값
# 회기 모델일 때 mae, mse를 쓴다
# 분류 모델일 때 (이진 일 때) binary_crossentropy를 쓴다 


loss= model.evaluate(x_test, y_test, batch_size=8)
# 지표를 만들기 위한 프레딕트 
print(loss)

# 성능이 향상 된 것이 아니라 계산하는 지표를 올바르게 해서 성능 향상처럼 보인다



print(x_test[-5:-1])
y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# 원하는 프레딕을 나오게 하려면 if문으로 변환하여 나오도록 한다


# [0.17937254905700684, 0.9824561476707458]

y_predict=np.argmax(y_pred, axis=1)
print('예측값 : ', y_predict)