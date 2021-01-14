
import numpy as np

# 데이터
x = np_load = np.load('./samsung/x_data.npy')
y = np_load = np.load('./samsung/y_data.npy')
x_pred = np.load('./samsung/x_pred.npy')

x1_shape = x.shape[1]
x2_shape = x.shape[2]

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred = x_pred.reshape(1,x_pred.shape[0]* x_pred.shape[1])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x1_shape, x2_shape)
x_test = x_test.reshape(x_test.shape[0], x1_shape, x2_shape)

from tensorflow.keras.models import load_model

# 모델
model = load_model('./samsung/samsung_model.h5')



# 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

# 예상 결과
x_predict = x_pred.reshape(1,x1_shape,x2_shape)
y_predict = model.predict(x_predict)

print(y_predict)

# loss, mae:  2531887.5 1221.1883544921875
# RMSE:  1591.1909
# R2:  0.9657210086094726
# [[88653.445]]