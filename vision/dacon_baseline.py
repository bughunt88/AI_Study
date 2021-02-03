import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D

# 데이터 로드

train = pd.read_csv('../data/vision/train.csv')
submission = pd.read_csv('../data/vision/submission.csv')
pred = pd.read_csv('../data/vision/test.csv')
####################################################


# 이미지 확인
# idx = 319
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx,'digit']
# letter = train.loc[idx,'letter']

# plt.title('index : %i, Digit : %s, Letter ; %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()
####################################################

#1. DATA
# x
x = train.drop(['id', 'digit', 'letter'], axis=1).values
x = x.reshape(-1, 28, 28, 1)
x = x/255.
print(x.shape)  # (2048, 28, 28, 1)

# y
# 뭐 하는 거지? one-hot-encoding??
y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order) >> 0으로 초기화된 넘파이 배열 
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1
print(y.shape)  # (2048, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
print(x_train.shape, x_test.shape)      # (1638, 28, 28, 1) (410, 28, 28, 1)

# predict
x_pred = pred.drop(['id', 'letter'], axis=1).values
x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255.
print(x_pred.shape) # (20480, 28, 28, 1)

#2. Modeling
def modeling() : 
    model = Sequential()
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu',\
        input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

model = modeling()

#3. Compile, Train
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

# submission
submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
print(submission.head())
submission.to_csv('../data/vision/file/submission.csv', index = False)
