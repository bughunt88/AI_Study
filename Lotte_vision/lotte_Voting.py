from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier
# VotingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate,KFold


import numpy as np
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Dropout, Input


#데이터 지정 및 전처리
x = np.load("../data/lpd_competition/npy/train_data_x9.npy",allow_pickle=True)
x_pred = np.load('../data/lpd_competition/npy/predict_data9.npy',allow_pickle=True)
y = np.load("../data/lpd_competition/npy/train_data_y9.npy",allow_pickle=True)

# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 

y = to_categorical(y)



def bulid_model():
    
    mobile = ResNet101(include_top=False,weights='imagenet',input_shape=x.shape[1:])
    mobile.trainable = True
    a = mobile.output
    a = GlobalAveragePooling2D() (a)
    a = Flatten() (a)
    a = Dense(4048, activation= 'relu') (a)
    a = Dropout(0.2) (a)
    a = Dense(1000, activation= 'softmax') (a)
    model = Model(inputs = mobile.input, outputs = a)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model

model2 = bulid_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=bulid_model, verbose=1)


def bulid_model1():
    
    efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x.shape[1:])
    efficientnet.trainable = True
    a = efficientnet.output
    a = GlobalAveragePooling2D() (a)
    a = Flatten() (a)
    a = Dense(4048, activation= 'relu') (a)
    a = Dropout(0.2) (a)
    a = Dense(1000, activation= 'softmax') (a)
    model1 = Model(inputs = efficientnet.input, outputs = a)
    model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
        
    return model1

model3 = bulid_model1()
model3 = KerasClassifier(build_fn=bulid_model1, verbose=1)


def bulid_model2():
    
    efficientnet = MobileNet(include_top=False,weights='imagenet',input_shape=x.shape[1:])
    efficientnet.trainable = True
    a = efficientnet.output
    a = GlobalAveragePooling2D() (a)
    a = Flatten() (a)
    a = Dense(4048, activation= 'relu') (a)
    a = Dropout(0.2) (a)
    a = Dense(1000, activation= 'softmax') (a)
    model2 = Model(inputs = efficientnet.input, outputs = a)
    model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
        
    return model2

model4 = bulid_model2()
model4 = KerasClassifier(build_fn=bulid_model2, verbose=1)


# ensemble 할 model 정의
models = [
    ('ada', model2),
    ('bc', model3),
    ('etc',model4)
]

kfold = KFold(n_splits=5, shuffle=True)

# soft vote
soft_vote  = VotingClassifier(models, voting='soft')
soft_vote_cv = cross_validate(soft_vote, x, y, cv=kfold)
soft_vote.fit(x, y)


tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    pred = soft_vote.predict(x_pred)
    predictions.append(pred)

final_pred = np.mean(predictions, axis=0)

sub = pd.read_csv('../data/lpd_competition/sample.csv')
sub['prediction'] = np.argmax(final_pred,axis = 1)
sub.to_csv('../data/lpd_competition/sample_003.csv',index=False)
