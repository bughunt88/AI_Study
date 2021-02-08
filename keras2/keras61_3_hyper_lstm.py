
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.

# 2. 모델

def build_model(drop=0.5, optimizer='adam', node=32, activation='relu', lr = 0.01) :
    inputs = Input(shape=(28,28), name='input')
    x = LSTM(256, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

model2 = build_model()

def create_hyperparameters() :
    batches = [16, 32, 64]
    optimizers = ['rmsprop', 'adam', 'adadelta', 'sgd']
    dropout = [0.1, 0.2, 0.3, 0.4]
    node = [128, 64, 32]
    activation =['relu','elu','prelu', 'softmax']
    lr = [0.1, 0.05, 0.01, 0.005, 0.001]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, \
        "activation" : activation, "node" : node, "lr" : lr}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=2)

search.fit(x_train, y_train, verbose=1)

print("best_params : ", search.best_params_)         
print("best_estimator : ", search.best_estimator_)   
print("best_score : ", search.best_score_)           

acc = search.score(x_test, y_test)
print("Score : ", acc)