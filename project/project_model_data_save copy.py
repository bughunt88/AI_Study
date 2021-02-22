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
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

min_level_db = -100
 
def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


TRAIN_DATA_DIR = '../data/project/train/'
TEST_DATA_DIR = '../data/project/test/'

DIR_List = ['angry/','nomal/','sad/']

for index, d_list in enumerate(DIR_List):

    print(index)

    for filename in os.listdir(TRAIN_DATA_DIR+d_list):
        filename = normalize('NFC', filename)
        try:
            wav, sr = librosa.load(TRAIN_DATA_DIR + d_list + filename,sr=16000)
            
            
            S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=512, n_mels=128) 
            
            log_S = librosa.power_to_db(S, ref=np.max)

            min_level_db = -100
            
            def _normalize(S):
                return np.clip((S - min_level_db) / -min_level_db, 0, 1)

            norm_S = _normalize(log_S)

            trainset.append((norm_S, index))

        except Exception as e:
            print(filename, e)
            raise





for index, d_list in enumerate(DIR_List):
    
    print(index)

    for filename in os.listdir(TEST_DATA_DIR+d_list):
        filename = normalize('NFC', filename)
        try:
            wav, sr = librosa.load(TEST_DATA_DIR + d_list + filename,sr=16000)
            
         
            S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=512, n_mels=128) 
            
            log_S = librosa.power_to_db(S, ref=np.max)

            min_level_db = -100
            
            def _normalize(S):
                return np.clip((S - min_level_db) / -min_level_db, 0, 1)

            norm_S = _normalize(log_S)

            testset.append((norm_S, index))

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
            
            S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=512, n_mels=128) 
            
            log_S = librosa.power_to_db(S, ref=np.max)

            min_level_db = -100
            
            def _normalize(S):
                return np.clip((S - min_level_db) / -min_level_db, 0, 1)

            norm_S = _normalize(log_S)

            kfoldset.append((norm_S, index))

        except Exception as e:
            print(filename, e)
            raise



# 학습 데이터를 무작위로 섞는다.
# random.shuffle(trainset)
random.shuffle(testset)
random.shuffle(trainset)
random.shuffle(kfoldset)

np.save('../data/project/data/kfold_data1.npy', arr=kfoldset)
np.save('../data/project/data/train_data1.npy', arr=trainset)
np.save('../data/project/data/test_data1.npy', arr=testset)




