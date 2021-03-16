import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1, # 수평 이동
    height_shift_range=0.1, # 수직 이동
    zoom_range=1.2, # 확대
    fill_mode='nearest' # 빈자리는 근처에 있는 것으로(padding='same'과 비슷)
)

test_datagen = ImageDataGenerator(rescale=1./255)


# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(150,150),
    batch_size=160,
    class_mode='binary'
)
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/LPD_competition/test',
    target_size=(150,150), # 리사이징 가능
    batch_size=120,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.

np.save('../data/LPD_competition/npy/kerass66_train_x.npy', arr=xy_train[0][0])
np.save('../data/lpd_competition/npy/kerass66_train_y.npy', arr=xy_train[0][1])
np.save('../data/lpd_competition/npy/kerass66_test_x.npy', arr=xy_test[0][0])
np.save('../data/lpd_competition/npy/kerass66_test_y.npy', arr=xy_test[0][1])

'''
x_train = np.load('../data/lpd_competition/npy/kerass66_train_x.npy')
y_train = np.load('../data/lpd_competition/npy/kerass66_train_y.npy')
x_test = np.load('../data/lpd_competition/npy/kerass66_test_x.npy')
y_test = np.load('../data/lpd_competition/npy/kerass66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
'''