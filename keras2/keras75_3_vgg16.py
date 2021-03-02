from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(32,32,3))

vgg16.trainable = False
# 훈련을 시킨다

vgg16.summary()
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))


model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
#model.add(Dense(10, activation='softmax'))
model.add(Dense(1))

model.summary()


print("그냥 가중치의 수 : ",len(model.weights))
print("동결하기 전 훈련되는 가중치의 수 : ",len(model.trainable_weights))



####### 요기 하단 때문에 파일 분리 함 ###########

import pandas as pd 

pd.set_option('max_colwidth', -1)   
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)

