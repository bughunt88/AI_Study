import numpy as np
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, MaxPooling2D
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101, EfficientNetB5
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


#데이터 지정 및 전처리
x = np.load("../data/lpd_competition/npy/train_data_x.npy",allow_pickle=True)
x_pred = np.load('../data/lpd_competition/npy/predict_data.npy',allow_pickle=True)
y = np.load("../data/lpd_competition/npy/train_data_y.npy",allow_pickle=True)

# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 


idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=64, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
# test_generator = idg2.flow(x_pred)

mc = ModelCheckpoint('../data/lpd_competition/lotte_0317_3.h5',save_best_only=True, verbose=1)

# efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile.trainable = True
a = mobile.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(4048, activation= 'relu') (a)
a = Dropout(0.2) (a)
a = Dense(1000, activation= 'softmax') (a)
model = Model(inputs = mobile.input, outputs = a)

early_stopping = EarlyStopping(patience= 20)
lr = ReduceLROnPlateau(patience= 10, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 64,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
from tensorflow.keras.models import Sequential, load_model

model = load_model('../data/lpd_competition/lotte_0317_3.h5')

tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(x_pred,verbose=True)
    predictions.append(preds)

final_pred = np.mean(predictions, axis=0)

sub = pd.read_csv('../data/lpd_competition/sample.csv')
sub['prediction'] = np.argmax(final_pred,axis = 1)
sub.to_csv('../data/lpd_competition/sample_004.csv',index=False)
