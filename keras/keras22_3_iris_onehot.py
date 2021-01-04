
# sklearn을 이용하여 OneHotEncoder 해보기 

# 다중 분류 !!!!


import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터

# x, y = load_iris(return_X_y=True)
# 교육용 데이터에서 위 처럼 데이터 값을 불러올 수 있다



dataset = load_iris()
x = dataset.data
y = dataset.target

'''
print(dataset.DESCR)
print(dataset.feature_names)

'''

print(x.shape) # (150,4)
print(y.shape) # (150,)
print(x[:5])
print(y)



# x 에 대한 전처리는 무조건이지만 y 에 대한 인코딩은 다중 분류에서 한다


# ******* 원핫인코딩 OneHotEncoding

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
labels = encoder.transform(y)

# 2차원 데이터로 변환합니다. 
labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
y = oh_encoder.transform(labels).toarray()

print(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3,  random_state = 66 ) 




# print(x.shape) #(569,30)
# print(y.shape) #(569,)

# print(x[:5])
# print(y)


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
model.add(Dense(169, input_shape=(4,), activation='relu'))
model.add(Dense(169, activation='relu'))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(3, activation='softmax'))
# softmax 는  위에서 원 핫 인코딩을 해야한다 그러면 아웃풋 쉐이프는 변경되어야 한다 


print(y_train.shape)
print(y_val.shape)
print(y_test.shape)




# 3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# metrics에 acc도 쓸 수 있고 accuracy 로 쓸 쑤 있다


from tensorflow.keras.callbacks import EarlyStopping
eraly_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') # mode는 min,max,auto 있다
# loss 값이 10번 최하로 떨어지면 그 지점에서 멈춘다 
# patience에 지정한 수 만큼 최저점이 지나가서 멈춘다 (보강을 해야한다)

model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val,y_val), callbacks=[eraly_stopping])


# loss의 값
# 회기 모델일 때 mae, mse를 쓴다
# 분류 모델일 때 (이진 일 때) binary_crossentropy를 쓴다 


loss= model.evaluate(x_test, y_test, batch_size=8)
# 지표를 만들기 위한 프레딕트 
print(loss)


print(x_test[-5:-1])
y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])



# 결과치 나오게 코딩할 것     # argmax

y_predict=np.argmax(y_pred, axis=1)
print('예측값 : ', y_predict)

