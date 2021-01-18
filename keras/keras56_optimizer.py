
# 러인 메이트 중요!!!

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


#optimizer = Adam(lr=0.1)
#loss :  5.8975692809326574e-05 결과물 :  [[10.983149]]
#optimizer = Adam(lr=0.001)
#loss :  4.0109531255438924e-05 결과물 :  [[11.013197]]
#optimizer = Adam(lr=0.0001)
#loss :  1.663002149143722e-05 결과물 :  [[10.992965]]

#optimizer = Adadelta(lr=0.1)
#loss :  0.001743039465509355 결과물 :  [[11.080125]]
#optimizer = Adadelta(lr=0.001)
#loss :  10.988611221313477 결과물 :  [[5.066224]]
#optimizer = Adadelta(lr=0.0001)
#loss :  25.0745792388916 결과물 :  [[2.1159678]]

#optimizer = Adamax(lr=0.1)
#loss :  208.50607299804688 결과물 :  [[25.001362]]
#optimizer = Adamax(lr=0.001)
#loss :  8.850379984437495e-09 결과물 :  [[10.999985]]
#optimizer = Adamax(lr=0.0001)
#loss :  0.001987927360460162 결과물 :  [[10.948274]]

#optimizer = Adagrad(lr=0.1)
#loss :  26653.96875 결과물 :  [[-139.59418]]
#optimizer = Adagrad(lr=0.001)
#loss :  3.487477897579083e-07 결과물 :  [[10.999272]]
#optimizer = Adagrad(lr=0.0001)
#loss :  0.005931953899562359 결과물 :  [[10.90331]]

#optimizer = RMSprop(lr=0.1)
#loss :  3820626.0 결과물 :  [[2269.2773]]
#optimizer = RMSprop(lr=0.001)
#loss :  0.4627138078212738 결과물 :  [[12.308347]]
#optimizer = RMSprop(lr=0.0001)
#loss :  0.4718744158744812 결과물 :  [[12.204898]]

#optimizer = SGD(lr=0.1)
#loss :  nan 결과물 :  [[nan]]
#optimizer = SGD(lr=0.001)
#loss :  3.390388201296446e-07 결과물 :  [[10.99881]]
#optimizer = SGD(lr=0.0001)
#loss :  0.0013612097827717662 결과물 :  [[10.95887]]

#optimizer = Nadam(lr=0.1)
#loss :  0.031003836542367935 결과물 :  [[11.334071]]
#optimizer = Nadam(lr=0.001)
#loss :  0.24667008221149445 결과물 :  [[9.932254]]
optimizer = Nadam(lr=0.0001)
#loss :  1.6811145542305894e-05 결과물 :  [[11.001647]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)
