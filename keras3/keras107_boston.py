# 회귀 모델적용
# pip install autokeras
# 두 방법으로 저장한 모델 비교


import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import  load_boston
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True, random_state = 66)

print(x_train.shape, y_train.shape) #(404, 13) (404,)
print(x_test.shape, y_test.shape)   #(102, 13) (102,)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#여러 모델 시도해보기==================================================================
'''
model = ak.ImageRegressor(
                            overwrite=True, 
                            max_trials=1,     
                            loss = 'mse',
                            metrics=['mae']   
)
# ValueError: Expect the data to ImageInput to have shape (batch_size, height, width, channels) or (batch_size, height, width) dimensions, but got input shape [32, 13]
# 데이터 형식: 3차원
'''
#============================
'''
model = ak.TextRegressor(
                            overwrite=True, 
                            max_trials=1,     
                            loss = 'mse',
                            metrics=['mae']   
)
# ValueError: Expect the data to TextInput to have shape (batch_size, 1), but got input shape [32, 13].
# 데이터 형식: 1차원
'''
#============================
# model = ak.StructuredDataRegressor(
#                             overwrite=True, 
#                             max_trials=2,     
#                             loss = 'mse',
#                             metrics=['mae'],
#                             project_name='boston_model',
#                             directory='./structured_data_regressor'  
# )
# 데이터 형식: 2차원
#==================================================================


# es =EarlyStopping(monitor='val_loss', mode='min', patience=6)
# re = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose = 2)
# mo = ModelCheckpoint('../data/keras3/modelcheckpoint/', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)

# model.fit(x_train, y_train, epochs=10, validation_split=0.2)#, callbacks = [es, re, mo]) # validation_splitr 기본 0.2

model = load_model('./structured_data_regressor/boston_model/best_model')    #[81.2420654296875, 7.692100524902344]

results = model.evaluate(x_test, y_test)

print(results)  

#============================
# best_model = model.tuner.get_best_model()
# best_model.save('../data/keras3/save/keras107_boston.h5', save_format='tf')
# NotImplementedError: Save or restore weights that is not an instance of `tf.Variable` is not supported in h5, use `save_format='tf'` instead. Got a model or layer MultiCategoryEncoding 
# with weights [<tensorflow.python.keras.engine.base_layer_utils.TrackableWeightHandler object at 0x000002843C09BF40>, 
# <tensorflow.python.keras.engine.base_layer_utils.TrackableWeightHandler object at 0x000002843C09B880>]
#============================