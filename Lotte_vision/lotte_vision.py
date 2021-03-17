import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input


#0. 변수
batch = 16
seed = 42
dropout = 0.2
epochs = 1000
model_path = '../data/LPD_competition/data/lpd_001.hdf5'
sub = pd.read_csv('../data/LPD_competition/sample.csv', header = 0)


#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    '../data/LPD_competition/train',
    target_size = (256, 256),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../data/LPD_competition/train',
    target_size = (256, 256),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    '../data/LPD_competition/test/',
    target_size = (256, 256),
    class_mode = None,
    batch_size = batch,
    seed = seed,
    shuffle = False
)

print(test_data.next())

np.save('../data/LPD_competition/npy/train_data_x.npy', arr=train_data[0][0])
np.save('../data/lpd_competition/npy/train_data_y.npy', arr=train_data[0][1])
np.save('../data/lpd_competition/npy/val_data_x.npy', arr=val_data[0][0])
np.save('../data/lpd_competition/npy/val_data_y.npy', arr=val_data[0][1])
np.save('../data/lpd_competition/npy/test_data_x.npy', arr=test_data[0][0])
np.save('../data/lpd_competition/npy/test_data_y.npy', arr=test_data[0][1])

'''
x_train = np.load('../data/lpd_competition/npy/kerass66_train_x.npy')
y_train = np.load('../data/lpd_competition/npy/kerass66_train_y.npy')
x_test = np.load('../data/lpd_competition/npy/kerass66_test_x.npy')
y_test = np.load('../data/lpd_competition/npy/kerass66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
'''