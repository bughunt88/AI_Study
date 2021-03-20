import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#0. 변수
batch = 256
seed = 42
dropout = 0.2
epochs = 10
model_path = '../data/LPD_competition/data/lpd_001.hdf5'
sub = pd.read_csv('../data/LPD_competition/sample.csv', header = 0)
es = EarlyStopping(patience = 5)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

x_train = np.load('../data/lpd_competition/npy/train_data_x.npy')
y_train = np.load('../data/lpd_competition/npy/train_data_y.npy')
x_test = np.load('../data/lpd_competition/npy/predict_data.npy')


y_train = to_categorical(y_train)

#2. 모델
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet
# efficientnetb7 = EfficientNetB7(include_top=False,weights='imagenet',input_shape=(256, 256, 3))
# efficientnetb7.trainable = False
mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
top_model = mobile_net.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Flatten()(top_model)
top_model = Dense(1024, activation="relu")(top_model)
top_model = Dense(1024, activation="relu")(top_model)
top_model = Dense(512, activation="relu")(top_model)
top_model = Dense(1000, activation="softmax")(top_model)

model = Model(inputs=mobile_net.input, outputs = top_model)
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
efficientnetb7 = EfficientNetB7(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
efficientnetb7.trainable = False
a = efficientnetb7.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(2028, activation="relu")(a)
#a = Dense(1024, activation="relu")(a)
#a = Dense(512, activation="relu")(a)
a = Dense(1000, activation="softmax")(a)

model = Model(inputs = efficientnetb7.input, outputs = a)

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=60, validation_split=0.2, callbacks = [es, cp, lr])

model = load_model(model_path)

#4. 평가 예측
pred = model.predict(x_test)
pred = np.argmax(pred, 1)
sub.loc[:,'prediction'] = pred
sub.to_csv('../data/lpd_competition/sample_001.csv', index = False)