# 실습
# 남자 여자 구별
# ImageDataGenerator의 fit 사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

'''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    vertical_flip= True,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    rotation_range= 5,
    zoom_range= 0.5,
    shear_range= 0.7,
    fill_mode = 'nearest',   
    validation_split=0.2505
 )

test_datagen = ImageDataGenerator(rescale=1./255)   #test data는 따로 튜닝하지 않고 전처리만 해준다.



xy_train = train_datagen.flow_from_directory(
     '../data/image/sex/',        
     target_size = (150,150),
     batch_size= 3000,  
     class_mode='binary', 
     subset = 'training'

 )

xy_test = train_datagen.flow_from_directory(
     '../data/image/sex/',       
     target_size = (150,150),
     batch_size= 3000,
     class_mode='binary', 
     subset = 'validation'
)


print(xy_train[0][0].shape) # (14, 150, 150, 3)
print(xy_train[0][1].shape) # (14,)

np.save('../data/image/brain/numpy/keras67_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/numpy/keras67_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/numpy/keras67_test_x.npy', arr=xy_test[0][0])
np.save('../data/image/brain/numpy/keras67_test_y.npy', arr=xy_test[0][1])

'''

x_train = np.load('../data/image/brain/numpy/keras67_train_x.npy')
y_train = np.load('../data/image/brain/numpy/keras67_train_y.npy')
x_test = np.load('../data/image/brain/numpy/keras67_test_x.npy')
y_test = np.load('../data/image/brain/numpy/keras67_test_y.npy')


model = Sequential()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(3))
model.add(Conv2D(32,3, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor= 'val_loss', patience=120)
lr = ReduceLROnPlateau(monitor='val_loss', patience=60, factor=0.5)
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['acc'])
history = model.fit(x_train,y_train, epochs=500, validation_data=(x_test,y_test),
callbacks=[es,lr])

loss=model.evaluate(x_test,y_test, batch_size=16)
print(loss)


# male_female(fit)
# loss :  2.4603981971740723
# acc :  0.5898617506027222
