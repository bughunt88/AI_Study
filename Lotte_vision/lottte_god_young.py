import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

#데이터 지정 및 전처리
x = np.load("../data/lpd_competition/npy/train_data_x.npy",allow_pickle=True)
x_pred = np.load('../data/lpd_competition/npy/predict_data.npy',allow_pickle=True)
y = np.load("../data/lpd_competition/npy/train_data_y.npy",allow_pickle=True)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred) 

idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),   # 0.1 => acc 하락
    height_shift_range=(-1,1),  # 0.1 => acc 하락
    # rotation_range=40, acc 하락 
    shear_range=0.2)

idg2 = ImageDataGenerator()

#y = np.argmax(y, axis=1)
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=32)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred

mc = ModelCheckpoint('../data/lpd_competition/lotte_0317_2.h5',save_best_only=True, verbose=1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet
mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
top_model = mobile_net.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Flatten()(top_model)
top_model = Dense(4000, activation="relu")(top_model)
# top_model = Dense(1024, activation="relu")(top_model)
# top_model = Dense(512, activation="relu")(top_model)
top_model = Dense(1000, activation="softmax")(top_model)
    
model = Model(inputs=mobile_net.input, outputs = top_model)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 20)
lr = ReduceLROnPlateau(patience= 10, factor=0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), 
                loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

learning_history = model.fit_generator(train_generator,epochs=200, 
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
model.load_weights('../data/lpd_competition/lotte_0317_2.h5')
#result = model.predict(x_pred,verbose=True)

tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(x_pred,verbose=True)
    predictions.append(preds)

final_pred = np.mean(predictions, axis=0)

sub = pd.read_csv('../data/lpd_competition/sample.csv')
sub['prediction'] = np.argmax(final_pred,axis = 1)
sub.to_csv('../data/lpd_competition/sample_006.csv',index=False)
