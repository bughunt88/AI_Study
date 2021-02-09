import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기 
    vertical_flip=True, # 수직 뒤집기 
    width_shift_range=0.1, # 수평 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=5, # 회전 
    zoom_range=1.2, # 확대
    shear_range=0.7, # 층 밀리기 강도?
    fill_mode='nearest' # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
)


test_datagen = ImageDataGenerator(rescale=1./255)


# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150), # 리사이징 가능
    batch_size=5,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.

model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3),padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
 
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit_generator(xy_train, steps_per_epoch=32, epochs=200, validation_data=xy_test, validation_steps=4)

# steps_per_epoch에 전체 데이터에 베치 사이즈 나눈 값을 적어야 한다 

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화 할 것 !!!

print("acc : ", acc[-1])
print("val_acc : ", val_acc[:-1])



####시각화
import matplotlib.pyplot as plt

plt.rc('font', family='NanumGothic') # For Windows

plt.figure(figsize=(10,6)) #가로세로 #이거 한 번 만 쓰기
plt.subplot(2,1,1)   #서브면 2,1,1이면 2행 1열 중에 1번째라는 뜻
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid() #바탕을 grid모눈종이로 하겠다

plt.title('손실비용') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(acc, marker='.', c='red')
plt.plot(val_acc, marker='.', c='blue')
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend('accuracy','val_accuracy')

plt.show()
