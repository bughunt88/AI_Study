import numpy as np
from sklearn.datasets import load_breast_cancer


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  test_size=0.3, random_state = 66 ) # shuffle False면 섞는다


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) 
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) 
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1) 


# 2. 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(120, activation='relu', input_shape=(30,1)))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1, activation='sigmoid')) 


# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
modelpath = '../data/modelcheckpoint/k46_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val,y_val), callbacks=[es,cp])


loss, mae = model.evaluate(x_test, y_test, batch_size=8)



y_predict = model.predict(x_val)


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
plt.plot(hist.history['accuracy'],marker='.',c='red',label='accuracy')
plt.plot(hist.history['val_accuracy'],marker='.',c='blue',label='accuracy')
plt.grid()

# plt.title('정확도')
plt.title('Accuracy')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()