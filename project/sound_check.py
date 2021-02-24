# -*- coding: utf-8 -*-
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

import warnings

warnings.filterwarnings('ignore')

min_level_db = -100
 
def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

DATA_DIR = '../data/project/predict/'


# 6초
#filename = '분노 데이터1 입니다.m4a'
#(104448,)

filename = '평상시 데이터 입니다.m4a'


wav, sr = librosa.load(DATA_DIR + filename,sr=16000)

print(wav.shape)


mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=50, n_fft=400, hop_length=1600)
#mfcc = sklearn.preprocessing.scale(mfcc, axis=1)




print(mfcc)



print(mfcc.shape)

mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

padded_mfcc = pad2d(mfcc, 650)

print(padded_mfcc.shape)


import matplotlib.pyplot as plt
import librosa.display
librosa.display.specshow(padded_mfcc, sr=16000, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

#(50, 31) 
#(50, 650)