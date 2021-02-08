
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. 데이터 / 전처리

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def build_model(drop=0.5, optimizer='adam', activation='relu') :
    inputs = Input(shape = (13,), name='input')
    x = Dense(16, activation=activation, name='hidden1')(inputs)
    x = Dense(16, activation=activation, name='hidden2')(x)
    x = Dense(32, activation=activation, name='hidden3')(x)
    x = Dense(32, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation=activation, name='hidden6')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model

def create_hyperparameters() :
    batches = [2, 4, 8, 16]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    activation =['relu','elu','prelu']
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "activation" : activation}

model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=100)   

path = '../data/modelcheckpoint/k62_5_{epoch:02d}_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.4, patience=3, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=2)

search.fit(x_train, y_train, verbose=1, validation_split=0.2, callbacks=[es, lr, cp])

print("best_params : ", search.best_params_)         
print("best_estimator : ", search.best_estimator_)   
print("best_score : ", search.best_score_)           
acc = search.score(x_test, y_test)
print("Score : ", acc)