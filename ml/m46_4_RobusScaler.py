# RobustScaler 는 이상치 제거 한 스케일러 
# 중위수로 계산한다
# StandardScaler 보다 이상치 정리 된 스케일링 
# 100프로 신뢰하지 말 것 !!!
# 아웃라이어 처리에 효과적이다 

# k18-6 가져와서 씀
# 스케일러중 로버스트스케일러를 써보자!
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

#데이터 전처리 (MinMax) 
print(np.max(x[0]))\

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# RobustScaler : 중위값을 기준으로 한다!
# 즉 이상치 제거를 하지 않은 상태에서 스탠다스 스케일러보다 성능이 좋다.
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x) 

# 스탠다스스케일러 적용
print(np.max(x), np.min(x)) #9.933930601860268 -3.9071933049810337
print(np.max(x[0]))         #0.44105193260704206


#트레인이랑 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#모델 짜기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(13,)) #input_dim=13 이랑 input_shape=(13,) 과도 같다.
dense1 = Dense(56, activation='relu')(inputs)
dense1 = Dense(28, activation='relu')(inputs)
dense1 = Dense(56, activation='relu')(inputs)
dense1 = Dense(28, activation='relu')(dense1)
dense1 = Dense(56, activation='relu')(dense1)
outputs = Dense(1, activation='relu')(dense1)

model = Model(inputs= inputs, outputs = outputs)

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=2, validation_split=0.2, verbose=1)

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss, mae : ', loss, mae)

y_predict1 = model.predict(x_test)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict1))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict1)
print('R2: ', R2)


