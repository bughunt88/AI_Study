# cv_results 해볼 것
import numpy as np
from tensorflow.keras.models import Sequential, Model
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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train,y_train, verbose=1)


print("#################################")

print(search.best_params_)

print(search.best_score_)

print("#################################")

print(search.cv_results_)

acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)


'''
RandomizedSearchCV

{'mean_fit_time': array([4.46005845, 1.40968911, 2.30957429, 1.43887218, 2.16853078,
       1.40688976, 1.40171456, 2.38489509, 2.36287562, 1.33070993]), 'std_fit_time': array([0.18518168, 0.05530231, 0.07117437, 0.09522011, 0.10622571,
       0.06579062, 0.02341734, 0.0845511 , 0.07373148, 0.0617219 ]), 'mean_score_time': array([1.87994345, 0.51355704, 0.89790217, 0.51765434, 0.80492338,
       0.48251247, 0.52580484, 0.97398901, 1.01631427, 0.43432673]), 'std_score_time': array([0.16419566, 0.00938201, 0.01069256, 0.00149924, 0.02209711,
       0.04715739, 0.00801322, 0.09256814, 0.13105162, 0.00422023]), 'param_optimizer': masked_array(data=['adadelta', 'adam', 'adam', 'adadelta', 'adam',
                   'rmsprop', 'adam', 'adadelta', 'adam', 'rmsprop'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.2, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.1, 0.2],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[10, 40, 20, 40, 30, 50, 40, 20, 20, 50],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 10}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 20}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 40}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 40}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}], 'split0_test_score': array([0.2696    , 0.94550002, 0.94580001, 0.11635   , 0.94585001,
       0.9368    , 0.94669998, 0.19194999, 0.95050001, 0.93769997]), 'split1_test_score': array([0.2816    , 0.93664998, 0.94630003, 0.11715   , 0.94559997,
       0.93790001, 0.94384998, 0.20895   , 0.95115   , 0.93889999]), 'split2_test_score': array([0.1596    , 0.94515002, 0.94854999, 0.1648    , 0.94894999,
       0.93760002, 0.94524997, 0.17995   , 0.95015001, 0.94505   ]), 'mean_test_score': array([0.23693334, 0.94243334, 0.94688334, 0.13276667, 0.94679999,
       0.93743334, 0.94526664, 0.19361666, 0.95060001, 0.94054999]), 'std_test_score': array([0.05490193, 0.00409195, 0.00119605, 0.02265334, 0.0015237 ,
       0.00046428, 0.00116357, 0.01189771, 0.00041432, 0.00321948]), 'rank_test_score': array([ 8,  5,  2, 10,  3,  7,  4,  9,  1,  6])}
'''


'''
GridSearchCV

{'mean_fit_time': array([5.01437044, 4.35964926, 4.4304107 , 4.84857551, 4.15174826,
       4.30187241, 5.11051718, 4.40605537, 4.33036049, 2.5417877 ,
       2.34696436, 2.36714236, 2.54401898, 2.33570218, 2.43550857,
       2.61658025, 2.38023885, 2.37071331, 2.41609653, 2.2177465 ,
       2.10685984, 2.24923197, 2.11825816, 2.11156368, 2.32856727,
       2.16951815, 2.13858167, 1.54734476, 1.3786335 , 1.48892442,
       1.5338459 , 1.49129423, 1.40543024, 1.59608452, 1.41420595,
       1.39205909, 1.33449181, 1.27634374, 1.21867434, 1.41476123,
       1.16986283, 1.26852695, 1.33868655, 1.29840509, 1.17031805]), 'std_fit_time': array([0.170573  , 0.10742101, 0.19581671, 0.30888992, 0.14092838,
       0.06165257, 0.03755176, 0.11938381, 0.05777003, 0.02048926,
       0.06205932, 0.07620252, 0.00988491, 0.09613292, 0.07340854,
       0.11865381, 0.05793805, 0.00757705, 0.10065425, 0.00719596,
       0.10823592, 0.08562305, 0.02471693, 0.16547902, 0.14062145,
       0.00761896, 0.09569874, 0.02829055, 0.01459913, 0.1209588 ,
       0.02285925, 0.08758748, 0.02033787, 0.105404  , 0.01779465,
       0.01267792, 0.03061388, 0.11809777, 0.02911984, 0.12520983,
       0.00640262, 0.1340077 , 0.0509466 , 0.11953766, 0.00497076]), 'mean_score_time': array([1.78054667, 1.80915062, 1.84459599, 1.76543125, 1.71255501,
       1.65999667, 1.78117871, 1.71773728, 1.70312214, 0.93924848,
       0.91851632, 0.90981189, 0.90140454, 0.89394569, 0.9486173 ,
       0.90385071, 0.93147953, 1.01290321, 0.85677218, 0.8490479 ,
       0.81353394, 0.80582166, 0.8336482 , 0.80786228, 0.83634122,
       0.82738439, 0.84456642, 0.59865069, 0.52201072, 0.53479783,
       0.52496163, 0.52596919, 0.51692875, 0.52373608, 0.53771178,
       0.59108575, 0.46469347, 0.44669056, 0.44032915, 0.44265922,
       0.43156052, 0.43548489, 0.45337431, 0.44127329, 0.42672976]), 'std_score_time': array([0.08617737, 0.07513761, 0.14587684, 0.10132043, 0.06777794,
       0.01975855, 0.00595891, 0.01284343, 0.01318834, 0.06481107,
       0.00409547, 0.01747091, 0.00711014, 0.00754864, 0.02110849,
       0.0044206 , 0.01874755, 0.07433733, 0.005335  , 0.00218275,
       0.00915868, 0.00526087, 0.0265864 , 0.0064032 , 0.03835666,
       0.00399249, 0.00372923, 0.08291568, 0.00785594, 0.00339179,
       0.00548505, 0.02213733, 0.0081597 , 0.01301101, 0.01710304,
       0.12271199, 0.02329723, 0.00692041, 0.01363448, 0.00635313,
       0.01136208, 0.01059594, 0.02057908, 0.01577946, 0.0070024 ]), 'param_batch_size': masked_array(data=[10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20,
                   20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40,
                   40, 40, 40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50,
                   50, 50, 50],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1,
                   0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2,
                   0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2,
                   0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3,
                   0.3],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_optimizer': masked_array(data=['rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
                   'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
                   'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta',
                   'rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
                   'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
                   'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta',
                   'rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
                   'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
                   'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'batch_size': 10, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 
0.2, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'rmsprop'}, 
{'batch_size': 30, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adadelta'}], 'split0_test_score': array([0.94704998, 0.95705003, 0.29769999, 0.9472    , 0.95324999,
       0.1364    , 0.94255   , 0.95365   , 0.18995   , 0.94875002,
       0.94725001, 0.1723    , 0.94814998, 0.95104998, 0.18545   ,
       0.94344997, 0.94599998, 0.21445   , 0.94679999, 0.949     ,
       0.18089999, 0.94905001, 0.94564998, 0.2439    , 0.94445002,
       0.94534999, 0.1934    , 0.94475001, 0.94319999, 0.13950001,
       0.94239998, 0.94139999, 0.14825   , 0.94150001, 0.9436    ,
       0.17625   , 0.94655001, 0.94475001, 0.17380001, 0.94120002,
       0.94284999, 0.1548    , 0.93965   , 0.94164997, 0.1166    ]), 'split1_test_score': array([0.94674999, 0.95555001, 0.27814999, 0.94615   , 0.95050001,
       0.16345   , 0.94279999, 0.94685   , 0.1752    , 0.94695002,
       0.94550002, 0.16785   , 0.94395   , 0.94520003, 0.18449999,
       0.94270003, 0.94814998, 0.22695   , 0.94395   , 0.94365001,
       0.1207    , 0.93150002, 0.94160002, 0.1663    , 0.94155002,
       0.94145   , 0.26365   , 0.94129997, 0.94265002, 0.13600001,
       0.94375002, 0.9429    , 0.12795   , 0.93889999, 0.94010001,
       0.08385   , 0.94075   , 0.94064999, 0.12995   , 0.94145   ,
       0.94029999, 0.15525   , 0.93814999, 0.93774998, 0.14915   ]), 'split2_test_score': array([0.95165002, 0.95060003, 0.25295001, 0.94985002, 0.95490003,
       0.25285   , 0.94714999, 0.95050001, 0.2173    , 0.94835001,
       0.94854999, 0.18880001, 0.94964999, 0.9515    , 0.19755   ,
       0.94354999, 0.95034999, 0.16575   , 0.94809997, 0.95085001,
       0.1797    , 0.94365001, 0.94335002, 0.1585    , 0.94400001,
       0.94520003, 0.11715   , 0.94515002, 0.94679999, 0.18275   ,
       0.94485003, 0.94164997, 0.11935   , 0.94225001, 0.94550002,
       0.1101    , 0.94305003, 0.94340003, 0.1034    , 0.94365001,
       0.94234997, 0.17065001, 0.93954998, 0.94104999, 0.1355    ]), 'mean_test_score': array([0.94848333, 0.95440002, 0.27626666, 0.94773334, 0.95288334,
       0.18423333, 0.94416666, 0.95033334, 0.19415   , 0.94801668,
       0.9471    , 0.17631667, 0.94724999, 0.94925   , 0.18916667,
       0.94323333, 0.94816665, 0.20238333, 0.94628332, 0.94783334,
       0.16043333, 0.94140001, 0.94353334, 0.18956667, 0.94333335,
       0.94400001, 0.1914    , 0.94373333, 0.94421667, 0.15275001,
       0.94366668, 0.94198332, 0.13185   , 0.94088334, 0.94306668,
       0.1234    , 0.94345001, 0.94293334, 0.13571667, 0.94210001,
       0.94183332, 0.16023333, 0.93911666, 0.94014998, 0.13375   ]), 'std_test_score': array([0.00224254, 0.0027559 , 0.01831757, 0.00155689, 0.00181491,
       0.04976016, 0.002112  , 0.00277859, 0.01744195, 0.00077172,
       0.00124965, 0.00901206, 0.00241246, 0.00286965, 0.00594059,
       0.00037931, 0.00177592, 0.02640156, 0.00173316, 0.00305296,
       0.02809998, 0.00733927, 0.00165846, 0.03855121, 0.00127432,
       0.00180417, 0.05982509, 0.00172838, 0.00184043, 0.02126127,
       0.00100196, 0.00065618, 0.01211638, 0.00143547, 0.00223656,
       0.03887679, 0.00238468, 0.00170605, 0.0290285 , 0.00110076,
       0.00110328, 0.00736799, 0.00068476, 0.00171464, 0.01334597]), 'rank_test_score': array([ 5,  1, 31,  9,  2, 37, 14,  3, 33,  7, 11, 38, 10,  4, 36, 21,  6,
       32, 12,  8, 39, 27, 18, 35, 20, 15, 34, 16, 13, 41, 17, 25, 44, 28,
       22, 45, 19, 23, 42, 24, 26, 40, 30, 29, 43])}

'''


