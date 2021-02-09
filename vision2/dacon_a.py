
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import string

'''
alphabets = string.ascii_lowercase
alphabets = list(alphabets)
'''

# train 데이터
train = pd.read_csv('../data/vision2/mnist_data/test.csv')


# train.csv데이터도 가져와서 쓰게 할 것!


# *********************
# train 데이터 
# 1회에 쓰던 mnist 데이터 A를 모아서 128으로 리사이징 해준다 
# 알파벳 별로 모델을 만들 것!


train['y_train'] = 1

a_train = train.loc[train['letter']=='A']

# a_train = a_train.drop(['id','digit','letter'],1)
a_train = a_train.drop(['id','letter'],1)

x_train = a_train.to_numpy().astype('int32')[:,:-1] # (780, 784)
y_train = a_train.to_numpy()[:,-1] # (780,)

# 이미지 전처리 100보다 큰 것은 253으로 변환, 100보다 작으면 0으로 변환
x_train[100 < x_train] = 253
x_train[x_train < 100] = 0

x_train = x_train.reshape(-1,28,28,1)

# 발리데이션이 필요 없다
# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.7, random_state = 66 ) # shuffle False면 섞는다

# 128로 리사이징 
x_train = experimental.preprocessing.Resizing(128,128)(x_train)
#x_val = experimental.preprocessing.Resizing(128,128)(x_val)



# *********************
# test 데이터
# 이번 대회에 주어진 50000개를 test 데이터로 사용해 정확한 모델을 만든다


# 코랩 데이터 
# x_data = np.load('/content/drive/My Drive/mnist/train_data.npy')
# y_data = pd.read_csv('/content/drive/My Drive/mnist/dirty_mnist_2nd_answer.csv')

# 컴터 데이터 
x_data = np.load('../data/vision2/train_data.npy')
y_data = pd.read_csv('../data/vision2/dirty_mnist_2nd_answer.csv')

y_test = y_data.to_numpy()[:,1] 

x_data = x_data.reshape(-1,256,256,1)

# 128로 리사이징 
x_data = experimental.preprocessing.Resizing(128,128)(x_data)
x_data = x_data.numpy().astype('int32')

# 이미지 전처리 253 보다 낮은 것은 0으로 변환 (위에서 253으로 지정했기 때문에 253 이상으로)
x_data[x_data < 253] = 0

x_test = x_data/255.0
x_train = x_train/255.0

# ImageDataGenerator의 값은 더 찾아볼 것!
idg = ImageDataGenerator( 
    
    horizontal_flip=True, # 수평 뒤집기 
    vertical_flip=True, # 수직 뒤집기 
    width_shift_range=0.1, # 수평 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=60, # 회전 
    zoom_range=[0.2,2] # 확대
    
    )
idg2 = ImageDataGenerator()



# *********************
# predict 데이터 넣을 예정
# test_dirty_mnist_2nd 5000개를 x_predict로 사용
# 각 알파벳 별로 0,1을 뽑는다 !!!





train_generator = idg.flow(x_train, y_train, batch_size=8, seed=2020)
test_generator = idg2.flow(x_test, y_test)
# val_generator = idg2.flow(x_val, y_val)

model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,1),padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
#model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
#model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
 
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])

learning_history = model.fit_generator(train_generator, epochs=100)


loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)

print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')



# loss :  10.671449661254883
# acc :  0.45914000272750854


# 생각보다 값이 좋지 못 함 
# 훈련 데이터의 양은 예전보다 많아짐 대략 700개

# 이유 예상 

# 1. 이미지 전처리 문제 
# 2. ImageDataGenerator 파라미터값들 수정
# 3. 모델이 구림

# 하나하나 처리해볼 것
