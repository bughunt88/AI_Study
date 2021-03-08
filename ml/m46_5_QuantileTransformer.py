# QuantileTransformer 는 이상치 제거 한 스케일러 
# 1000개의 분위수 
# 이상치에 대해서 효과있는 방법
# 데이터 1000개 이상일 때 사용할 것
# 아웃라이어 처리에 효과적이다 


import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

#데이터 전처리 (MinMax) 
print(np.max(x[0]))\

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# QuantileTransformer는 RobustScaler와 비슷하지만 분위수 자체를 1000개로 잡아서 세밀하게 한다. 하지만 분위수를 이용하기 때문에 이상치 처리에 강력하다.
# RobustScaler, QuantileTransformer 둘을 이용하면 이상치(outlier)를 처리하지 않아도 되지만 그래도 하자!

scaler = QuantileTransformer()
# QuantileTransformer는 파라미터가 하나 있다!
'''
scaler = QuantileTransformer(output_distribution='normal')  # normal: 정규분포
scaler = QuantileTransformer() # 디폴트: 균등분포
'''

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