import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))     #보통 내가 불러올 모델들은 아웃풋빼고 히든부분까지만 되어있을 것이다.

model.summary()

# 모델 저장
# model.save('./model/save_keras35.h5')
# . > 현재 폴더라는 뜻 > 즉 STUDY라는 뜻
# 왼쪽 모델 폴더에 파일이 생성됨을 볼 수 있음

# 잘 만들어진 모델을 저장하여 불러와 쓸 수 있다. 혹은 대회에서 우승한 모델들을 가져와서 쓸 수도 있다.
#######################


# 모델 저장 슬라이싱
model.save('./model/save_keras35.h5')
model.save('.//model//save_keras35_1.h5')
model.save('.\model\save_keras35_2.h5')
model.save('.\\model\\save_keras35_3.h5')
# 모두 된다.

