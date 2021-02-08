

# 레이어를 변경해봤다 

# 돌려보자!


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

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

n_splits_num = 128

# cross validation
skf = StratifiedKFold(n_splits=n_splits_num, random_state=42, shuffle=True)

reLR = ReduceLROnPlateau(patience=100,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=160, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, test_index  in skf.split(train2,train['digit']) :
    
    mc = ModelCheckpoint('../data/vision/checkpoint/best_cvision.h5',save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_test = train2[test_index]
    y_train = train['digit'][train_index]
    y_test = train['digit'][test_index]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9, shuffle=True, random_state=47)

    train_generator = idg.flow(x_train, y_train, batch_size=16, seed=2020)
    test_generator = idg2.flow(x_test, y_test)
    valid_generator = idg2.flow(x_valid, y_valid)
    pred_generator = idg2.flow(test2, shuffle=False)
    
    dropout_rate=0.5
    model_in = Input(shape = (28,28,1))
    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(model_in) 
    # kernel_initializer : 레이어의 초기 난수 가중치를 설정하는 방식
    # he_normal : 0을 중심으로 stddev = sqrt(2 / fan_in)의 표준편차를 가진 절단된 정규분포에 따라 샘플이 생성
    x = BatchNormalization()(x)
    x_res = x
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation('relu')(x)
    x = AveragePooling2D()(x) # MaxPooling은 영역중 가장 큰 값을 뽑지만, AveragePooling은 지정한 영역에 평균 추출 방식
    x = Dropout(rate=dropout_rate)(x)

    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    # 256
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)
    ##

    x = Flatten()(x)

    # 256(ok) --> 512(X)
    x = Dense(units=256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(units=256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x_res, x])
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate)(x)

    model_out = Dense(units=10, activation='softmax')(x)
    model = Model(model_in, model_out)
    

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    
    learning_history = model.fit_generator(train_generator,epochs=2000, validation_data=valid_generator, callbacks=[es,mc,reLR])

    #4. Evaluate, Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)
    
    # predict
    result += model.predict_generator(pred_generator, verbose=True)/n_splits_num
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')
    

sub['digit'] = result.argmax(1)
sub.to_csv('../data/vision/file/submission8.csv', index = False)


print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')


# submission.csv