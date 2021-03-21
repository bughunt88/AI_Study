import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input, GaussianDropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

#0. 변수
filenum = 13
batch = 6
seed = 42
dropout = 0.4
epochs = 1000
model_path = load_model('../data/lpd_competition/lotte_0317_3.h5')

save_folder = '../data/lpd_competition/sample_006.csv'

sub = pd.read_csv('../data/lpd_competition/sample.csv', header = 0)

es = EarlyStopping(patience = 7)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    preprocessing_function= preprocess_input,
    horizontal_flip= True
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    width_shift_range= 0.05,
    height_shift_range= 0.05,
    horizontal_flip= True
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    '../data/lpd/train_new',
    target_size = (224, 224),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../data/lpd/train_new',
    target_size = (224, 224),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    '../data/lpd/test_new',
    target_size = (224, 224),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

#2. 모델
eff = EfficientNetB4(include_top = False, input_shape=(224, 224, 3))
eff.trainable = True

pretrained = eff.output
layer_1 = Flatten()(pretrained)
layer_2 = Dense(2000, activation = 'relu')(layer_1)
layer_2 = Dropout(dropout)(layer_2)
layer_3 = Dense(1000, activation = 'relu')(layer_2)
output = Dense(1000, activation = 'softmax')(layer_3)

model = Model(inputs = eff.input, outputs = output)

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])
model.fit(train_data, steps_per_epoch = np.ceil(39000/batch), validation_data= val_data, validation_steps= np.ceil(9000/batch),\
    epochs = epochs, callbacks = [es, cp, lr])

model = load_model(model_path)

#4. 평가 예측
result = []
'''
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - mode')
    pred = model.predict(test_data, steps = len(test_data))
    pred = np.argmax(pred, 1)
    result.append(pred)
    print(f'{tta+1} 번째 제출 파일 저장하는 중')
    temp = np.array(result)
    temp = np.transpose(result)
    temp_mode = stats.mode(temp, axis = 1).mode
    sub.loc[:, 'prediction'] = temp_mode
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)
    temp_count = stats.mode(temp, axis = 1).count
    for i, count in enumerate(temp_count):
        if count < tta/2.:
            print(f'{tta+1} 반복 중 {i} 번째는 횟수가 {count} 로 {(tta+1)/2.} 미만!')
'''
cumsum = np.zeros([72000, 1000])
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps = len(test_data)) # (72000, 1000)
    
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)
    
    count = 0
    print(f'TTA {tta} : {count} 개가 불확실!')
    i = 0
    for percent in temp_percent:
        count = 0
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'{tta+1} ')
    print(f'{tta+1} 번째 제출 파일 저장하는 중')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)
    