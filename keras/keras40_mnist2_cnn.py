# 인공지능계의 hello world라 불리는 mnist!!!


# 실습!!! 완성하시요!!!
# 지표는 acc

# 응용
# y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)

# 0.985





import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test,y_test)= mnist.load_data()

# plt.imshow(x_train[1])
# plt.show()



# 데이터 전처리

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))



# OnHotEncoding (다중 분류 데이터에서 Y값을 해주는 것)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train)
labels = encoder.transform(y_train)

labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
y_train = oh_encoder.transform(labels).toarray()


encoder = LabelEncoder()
encoder.fit(y_test)
labels = encoder.transform(y_test)

labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
y_test = oh_encoder.transform(labels).toarray()





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(100,  kernel_size=(2,2)) )
model.add(Conv2D(50,  kernel_size=(2,2)) )
model.add(Conv2D(10,  kernel_size=(2,2)) )
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=10, batch_size=500, validation_batch_size=0.2, callbacks=earlystopping)

loss, mae = model.evaluate(x_test, y_test, batch_size=500)


print(loss)
print(mae)

y_predict = model.predict(x_test[:10])


print('y_test : ',  y_test[:10].argmax(axis=1))
print('y_predict_argmax : ', y_predict.argmax(axis=1)) 


