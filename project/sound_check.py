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


frame_length = 0.025
frame_stride = 0.010


min_level_db = -100
 
def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

'''
def Mel_S(wav_file):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=16000)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))
    S_1 = librosa.power_to_db(S, ref=np.max)
    S = _normalize(S_1)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()

    return S
man_original_data = '../data/project/predict/060-126.m4a'
mel_spec = Mel_S(man_original_data)
'''

pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

DATA_DIR = '../data/project/predict/'

filename = '060-126.m4a'

wav, sr = librosa.load(DATA_DIR + filename)

mfcc = librosa.feature.mfcc(wav,sr=16000, n_mfcc=80, n_fft=1000, hop_length=160)

#mfcc = librosa.feature.mfcc(wav)


print(mfcc)
mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
print("#####")
print(mfcc)
print(np.max(mfcc))
print(np.min(mfcc))

padded_mfcc = pad2d(mfcc, 240)

import matplotlib.pyplot as plt
import librosa.display
librosa.display.specshow(padded_mfcc, sr=16000, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()