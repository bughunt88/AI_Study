# boston, 

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_wine
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10,cifar100

import numpy as np


boston_datasets = load_boston()
diabetes_datasets = load_diabetes()
cancer_datasets = load_breast_cancer()
wine_datasets = load_wine()


boston_x = boston_datasets.data
boston_y = boston_datasets.target

np.save('../data/npy/boston_x.npy', arr=boston_x)
np.save('../data/npy/boston_y.npy', arr=boston_y)    


diabetes_x = diabetes_datasets.data
diabetes_y = diabetes_datasets.target

np.save('../data/npy/diabetes_x.npy', arr=diabetes_x)
np.save('../data/npy/diabetes_y.npy', arr=diabetes_y)    


cancer_x = cancer_datasets.data
cancer_y = cancer_datasets.target

np.save('../data/npy/cancer_x.npy', arr=cancer_x)
np.save('../data/npy/cancer_y.npy', arr=cancer_y)    


wine_x = wine_datasets.data
wine_y = wine_datasets.target

np.save('../data/npy/wine_x.npy', arr=wine_x)
np.save('../data/npy/wine_y.npy', arr=wine_y)    


# 6. mnist
(m_x_train,m_y_train),(m_x_test,m_y_test) = mnist.load_data()
(fashion_x_train,fashion_y_train),(fashion_x_test,fashion_y_test) = fashion_mnist.load_data()
(cifar10_x_train,cifar10_y_train),(cifar10_x_test,cifar10_y_test) = cifar10.load_data()
(cifar100_x_train,cifar100_y_train),(cifar100_x_test,cifar100_y_test) = cifar100.load_data()


np.save('../data/npy/mnist_x_train.npy', arr=m_x_train)
np.save('../data/npy/mnist_y_train.npy', arr=m_y_train)    

np.save('../data/npy/mnist_x_test.npy', arr=m_x_test)
np.save('../data/npy/mnist_y_test.npy', arr=m_y_test)    

np.save('../data/npy/fashion_x_train.npy', arr=fashion_x_train)
np.save('../data/npy/fashion_y_train.npy', arr=fashion_y_train)    

np.save('../data/npy/fashion_x_test.npy', arr=fashion_x_test)
np.save('../data/npy/fashion_y_test.npy', arr=fashion_y_test)    

np.save('../data/npy/cifar10_x_train.npy', arr=cifar10_x_train)
np.save('../data/npy/cifar10_y_train.npy', arr=cifar10_y_train)    

np.save('../data/npy/cifar10_x_test.npy', arr=cifar10_x_test)
np.save('../data/npy/cifar10_y_test.npy', arr=cifar10_y_test)    

np.save('../data/npy/cifar100_x_train.npy', arr=cifar100_x_train)
np.save('../data/npy/cifar100_y_train.npy', arr=cifar100_y_train)    

np.save('../data/npy/cifar100_x_test.npy', arr=cifar100_x_test)
np.save('../data/npy/cifar100_y_test.npy', arr=cifar100_y_test)   



def load_data_npy(npy_location, check_num):


    if check_num == 1:

        check_data = dict(
        x_data = np.load('../data/npy/'+npy_location+'_x.npy'),
        y_data = np.load('../data/npy/'+npy_location+'_y.npy')
        )

    elif check_num == 2:
        
        check_data = dict(
        x_test = np.load('../data/npy/'+npy_location+'_x_test.npy'),
        y_test = np.load('../data/npy/'+npy_location+'_y_test.npy'),
        x_train = np.load('../data/npy/'+npy_location+'_x_train.npy'),
        y_train = np.load('../data/npy/'+npy_location+'_y_train.npy')
        )
    else:

        check_data = '(1 : sklearn 데이터 or 2 : mnist 데이터) 선택 하세요'

    return check_data





print(load_data_npy('cifar100',3))










