

# BatchNormalization 이랑 Drop out 이랑 같이 사용하지 않는다


# 해볼 것 
# 1. 리사이징 조금 크게 돌려보기 
# 2. https://dacon.io/competitions/official/235626/codeshare/1624  숫자 얻기 해서 적용시켜보기
# 4. swish 넣고 돌려보기 



# swish로 돌려 봄 

# 돌려야 함 


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

train = pd.read_csv('../data/vision/train.csv')
sub = pd.read_csv('../data/vision/submission.csv')
test = pd.read_csv('../data/vision/test.csv')

# drop columns
train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values

# reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

'''
# 리사이징
# (64, 64), (128, 128), (256, 256)
train2 = experimental.preprocessing.Resizing(24,24)(train2)
test2 = experimental.preprocessing.Resizing(24,24)(test2)

train2 = np.array(train2)
test2 = np.array(test2)
'''


# data normalization
train2 = train2/255.0
test2 = test2/255.0


# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()


n_splits_num = 16


# cross validation
skf = StratifiedKFold(n_splits=n_splits_num, random_state=42, shuffle=True)

reLR = ReduceLROnPlateau(patience=100,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=160, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(train2,train['digit']) :
    
    mc = ModelCheckpoint('../data/vision/checkpoint/best_cvision.h5',save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train,batch_size=16)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(test2,shuffle=False)
    
    model = Sequential()

    model.add(Conv2D(16,(3,3),activation='swish',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32,(3,3),activation='swish',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='swish',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='swish',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='swish',padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64,(3,3),activation='swish',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='swish',padding='same')) 
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128,activation='swish'))
    model.add(BatchNormalization())
 
    model.add(Dense(64,activation='swish'))
    model.add(BatchNormalization())

    model.add(Dense(10,activation='softmax'))
    

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    
    learning_history = model.fit_generator(train_generator,epochs=2000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('../data/vision/checkpoint/best_cvision.h5')
    result += model.predict_generator(test_generator,verbose=True)/n_splits_num
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')
    
sub['digit'] = result.argmax(1)
sub.to_csv('../data/vision/file/submission4.csv', index = False)


print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')


# submission4.csv
# 0.9215686275	

