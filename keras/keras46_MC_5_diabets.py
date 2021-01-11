
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 2. 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model=Sequential()
model.add(Dense(128, activation='relu', input_shape=(10,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
modelpath = './ModelCheckPoint/k46_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[es,cp])


# 4. 평가, 예측



loss, mae = model.evaluate(x_test, y_test, batch_size=8)
y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 



print('loss : ', loss)
print('mae : ', mae )


print("RMSE : ", RMSE(y_test, y_predict))

# R2 만드는 법
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)





# 시각화
import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()

#plt.title('손실비용')
plt.title('Cost Loss')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['mae'],marker='.',c='red',label='mae')
plt.plot(hist.history['val_mae'],marker='.',c='blue',label='mae')
plt.grid()

# plt.title('정확도')
plt.title('MAE')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()