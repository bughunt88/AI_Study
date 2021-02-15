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
'''
audio_path = '../data/project/001-101.m4a'
wav, sr = librosa.load(audio_path)
print("######################")
print(wav)
print(sr)
'''

trainset = []
testset = []

# split each set into raw data, mfcc data, and y data
# STFT 한 것, CNN 분석하기 위해 Spectogram으로 만든 것, MF한 것, mel0spectogram 한 것
train_X = []
train_mfccs = []
train_y = []

test_X = []
test_mfccs = []
test_y = []

# 모든 음성파일의 길이가 같도록 후위에 padding 처리
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i-a.shape[0])))
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

frame_length = 0.025
frame_stride = 0.0010

DATA_DIR = '../data/project/train/angry/'

# 0 - 화남 
for filename in os.listdir(DATA_DIR):
    filename = normalize('NFC', filename)
    try:
        # wav 포맷 데이터만 사용
        '''
        if '.wav' not in filename in filename:
            continue
        '''
        
        wav, sr = librosa.load(DATA_DIR + filename, sr=16000)
        
        mfcc = librosa.feature.mfcc(wav, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        padded_mfcc = pad2d(mfcc, 40)


        trainset.append((padded_mfcc, 0))
        
        '''
        # 추임새 별로 dataset에 추가
        if filename[0] == '어':
            trainset.append((padded_mfcc, 0))
        elif filename[0] == '음':
            trainset.append((padded_mfcc, 1))
        elif filename[0] == '그':
            trainset.append((padded_mfcc, 2))
        '''

    except Exception as e:
        print(filename, e)
        raise

# 학습 데이터를 무작위로 섞는다.
random.shuffle(trainset)

print(trainset)