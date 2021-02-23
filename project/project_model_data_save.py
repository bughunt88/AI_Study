import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import librosa
import sklearn
import random
from unicodedata import normalize
from keras.layers import Dense
from keras import Input
from keras.engine import Model
from keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D

import warnings

warnings.filterwarnings('ignore')

trainset = []
testset = []
kfoldset = []


# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

TRAIN_DATA_DIR = '../data/project/train/'
TEST_DATA_DIR = '../data/project/test/'

DIR_List = ['angry/','nomal/','sad/']

for index, d_list in enumerate(DIR_List):

    print(index)

    for filename in os.listdir(TRAIN_DATA_DIR+d_list):
        filename = normalize('NFC', filename)
        try:
            wav, sr = librosa.load(TRAIN_DATA_DIR + d_list + filename,sr=16000)
            mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=50, n_fft=400, hop_length=160)
            mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
            padded_mfcc = pad2d(mfcc, 650)
            trainset.append((padded_mfcc, index))

        except Exception as e:
            print(filename, e)
            raise





for index, d_list in enumerate(DIR_List):
    
    print(index)

    for filename in os.listdir(TEST_DATA_DIR+d_list):
        filename = normalize('NFC', filename)
        try:
            wav, sr = librosa.load(TEST_DATA_DIR + d_list + filename,sr=16000)
            mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=50, n_fft=400, hop_length=160)
            mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
            padded_mfcc = pad2d(mfcc, 650)
            testset.append((padded_mfcc, index))

        except Exception as e:
            print(filename, e)
            raise


TEST_DATA_DIR = '../data/project/kfold/'
DIR_List = ['angry/','nomal/','sad/']

for index, d_list in enumerate(DIR_List):
    
    print(index)

    for filename in os.listdir(TEST_DATA_DIR+d_list):
        filename = normalize('NFC', filename)
        try:
            wav, sr = librosa.load(TEST_DATA_DIR + d_list + filename,sr=16000)
            mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=50, n_fft=400, hop_length=160)
            mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
            padded_mfcc = pad2d(mfcc, 650)
            kfoldset.append((padded_mfcc, index))

        except Exception as e:
            print(filename, e)
            raise



# 학습 데이터를 무작위로 섞는다.
random.shuffle(testset)
random.shuffle(trainset)
random.shuffle(kfoldset)

np.save('../data/project/data/kfold_data3.npy', arr=kfoldset)
np.save('../data/project/data/train_data3.npy', arr=trainset)
np.save('../data/project/data/test_data3.npy', arr=testset)




