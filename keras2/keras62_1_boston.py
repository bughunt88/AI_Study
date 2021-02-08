import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV


dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.7, random_state = 77, shuffle=True ) 

# 1. 데이터 / 전처리


# 2. 모델

def bulid_model(drop=0.5, optimizer='adam',activation='relu'):
    
    inputs = Input(shape=(13), name='Input')
    x = Dense(512, activation=activation, name='Hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=optimizer, metrics='mae')
    
    return model


def create_hyperparameters():
    batches = [100,200]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2,0.3]
    activation = ['relu', 'selu', 'swish']
    epochs = [10,20,30]
    validation_batch_size = [0.1,0.2,0.3]
    return {'batch_size': batches, "optimizer":optimizers, "drop":dropout,  "activation":activation, "epochs":epochs, "validation_batch_size":validation_batch_size}
# 파라미터는 키 벨류 형태로 만든다 

hyperparameters = create_hyperparameters()
model2 = bulid_model()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasRegressor(build_fn=bulid_model, verbose=1)
# Keras로 래핑해서 ML에서도 인식하도록 만든다


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

# ML에 랜덤 서치를 사용해서 DL이랑 묶는다


search.fit(x_train,y_train, verbose=1)


print("#################################")

print(search.best_params_)
# {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 10}
# 내가 파라미터 넣은 것 중에 뽑는다

# print(search.best_estimator_)
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000016DFEB8BE20>
# 전체에서 좋은걸 뽑는다 
# 래핑하면 잘 안나온다 사용하지 말자!

print(search.best_score_)
# 아래 점수랑 다르다

print("#################################")

'''
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)
'''