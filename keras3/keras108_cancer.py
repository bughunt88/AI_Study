

import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import  load_boston, load_breast_cancer
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)

print(x_train.shape, y_train.shape) #(455, 30) (455,)
print(x_test.shape, y_test.shape)   #(114, 30) (114,)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#============================
# model = ak.StructuredDataClassifier(
#                             overwrite=True, 
#                             max_trials=2,     
#                             loss = 'mse',
#                             # metrics=['acc'],
#                             # ValueError: Objective value missing in metrics reported to the Oracle, expected: ['val_accuracy'], found: dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
#                             project_name='cancer_model',
#                             directory='./structured_data_classifier'  
# )
# 데이터 형식: 2차원
#==================================================================


# es =EarlyStopping(monitor='val_loss', mode='min', patience=6)
# re = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose = 2)
# mo = ModelCheckpoint('../data/keras3/modelcheckpoint/', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)

# model.fit(x_train, y_train, epochs=10, validation_split=0.2)#, callbacks = [es, re, mo]) # validation_splitr 기본 0.2

model = load_model('./structured_data_classifier/cancer_model/best_model') # 계속 같은 경로에 저장하면 덮어쓰기 됨   
# [0.0881754532456398,  0.9736841917037964]
# [0.07780305296182632, 0.9736841917037964]
# [0.08008598536252975, 0.9824561476707458]
# [0.0797407478094101,  0.9561403393745422]***

results = model.evaluate(x_test, y_test)

print(results)  