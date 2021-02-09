import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten

# 하나의 폴더에서 데이터 train, test 나누기 
# ImageDataGenerator에 validation_split 준다 
# flow_from_directory에서 subset='training', subset='validation'을 써준다
# (validation_split을 준 곳에 같이 만들 것!!!)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2505
)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/sex',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/image/sex',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
# Found 1389 images belonging to 2 classes.
# Found 347 images belonging to 2 classes.
print(xy_train[0][0].shape) # (14, 150, 150, 3)
print(xy_train[0][1].shape) # (14,)


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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras62_1_checkpoint_{val_loss:.4f}-{epoch:02d}.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
history = model.fit_generator(xy_train, steps_per_epoch=93, epochs=500, validation_data=xy_val, validation_steps=31,
callbacks=[es])


'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs = len(acc)
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, acc, label='train')
ax.plot(x_axis, val_acc, label='val')
ax.legend()
plt.ylabel('acc')
plt.title('acc')
# plt.show()


fig, ax = plt.subplots()
ax.plot(x_axis, loss, label='train')
ax.plot(x_axis, val_loss, label='val')
ax.legend()
plt.ylabel('loss')
plt.title('loss')
plt.show()
'''

print("y 값")
print(xy_train[0][1])
print("y 계산 값")
print(model.predict_generator(xy_train[0][0], verbose=True))

y_predict = model.predict_generator(xy_train[0][0], verbose=True)

print(np.where(y_predict> 0.5, 1, 0))

acc = history.history['acc']

print("acc 값")
print(acc)

