
# load_model 체크 포인트 
# 체크포인트는 loss가 낮을 때 저장한 것이기 떄문에 가장 웨이트가 좋다 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 

#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model

model = load_model('../data/modelcheckpoint/k52_1_mnist_checkpoinrt.hdf5')

result=model.evaluate(x_test,y_test, batch_size=16)
print('로드체크포인트_loss : ', result[0])
print('로드체크포인트_acc : ', result[1])

# 로드체크포인트_loss :  0.07213222980499268
# 로드체크포인트_acc :  0.9775999784469604