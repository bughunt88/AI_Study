# 가중치 저장할 것
# 1. model.save() 쓸 것
# 2. pickle 쓸 것


import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)  
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


# 2. 모델

def bulid_model(drop=0.5, optimizer='adam'):
    
    inputs = Input(shape=(28*28,), name='Input')
    x = Dense(512, activation='relu', name='Hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    
    return model


def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2,0.3]
    return {'batch_size': batches, "optimizer":optimizers, "drop":dropout}

hyperparameters = create_hyperparameters()
model2 = bulid_model()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=bulid_model, verbose=1)

import pickle
pickle.dump(model2, open('../data/xgb_save/keras64.pickle.dat', 'wb'))
# 피클 위치!!!!!

model4 = pickle.load(open('../data/xgb_save/keras64.pickle.dat', 'rb'))
# 피클 로드


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train,y_train, verbose=1)

print("#################################")


search.best_estimator_.model.save('../data/h5/k64_modelbest.h5')
# 모델 세이브 위치!!!!!
# 모델 로드하고 스코어는 사용불가! 
# predict로 점수 낼 것 

# model6 = load_model('../data/h5/k64_modelbest.h5')
# y_predict = model6.predict(x_test)


print(search.best_params_)

print(search.best_score_)

print("#################################")

acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)



