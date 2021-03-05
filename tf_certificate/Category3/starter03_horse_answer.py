# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization,MaxPool2D, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import glob,numpy as np
from PIL import Image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

def solution_model():
    # _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    # _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    # urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    # local_zip = 'horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/horse-or-human/')
    # zip_ref.close()
    # urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    # local_zip = 'testdata.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('../data/image/test2/')
    # zip_ref.close()

    TRAINING_DIR = "../data/image/test2/"
    caltech_dir =  TRAINING_DIR
    categories = ['humans','horses'] 
    nb_classes = len(categories)

    image_w = 150
    image_h = 150

    pixels = image_h * image_w * 3

    x = []
    y = []

    for idx, cat in enumerate(categories):
        
        #one-hot 돌리기.
        label = [0 for i in range(nb_classes)]
        label[idx] = 1

        image_dir = caltech_dir + "/" + cat
        files = glob.glob(image_dir+"/*.png")
        print(cat, " 파일 길이 : ", len(files))
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)

            x.append(data)
            y.append(label)


    x = np.array(x)
    y = np.array(y)
    # #1 0 0 0 이면 Beagle
    # #0 1 0 0 이면 

    print(x.shape)
    print(y.shape)


    np.array(x)
    np.array(y)

    train_datagen = ImageDataGenerator(
        width_shift_range=(0.1),   #
        height_shift_range=(0.1),
        zoom_range= 0.05  
        )    

    validation_datagen = ImageDataGenerator()   
        
    #전처리
    x = preprocess_input(x)

    categories = ['humans','horses'] 
    nb_classes = len(categories)
    y = np.argmax(y,1)
    x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size= 0.2)
    # from sklearn.model_selection import StratifiedKFold, KFold
    # skf = StratifiedKFold(n_splits=8, random_state=42, shuffle=True)

    # nth = 0

    # for train_index, valid_index in skf.split(x,y) :  

    #     x_train = x[train_index]
    #     x_valid = x[valid_index]    
    #     y_train = y[train_index]
    #     y_valid = y[valid_index]

    train_generator = train_datagen.flow(x_train,y_train,batch_size=32)
    valid_generator = validation_datagen.flow(x_valid,y_valid)


    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size =(3,3), padding = 'valid', strides=(1,1),
    input_shape = (150,150,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),               
    tf.keras.layers.Conv2D(filters = 32, kernel_size =(3,3), padding = 'valid',strides=(1,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'), 
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),

    tf.keras.layers.Conv2D(filters = 64, kernel_size =(3,3), padding = 'valid',strides=(1,1)),
    tf.keras.layers.BatchNormalization() ,   
    tf.keras.layers.Activation('relu'),                         
    tf.keras.layers.Conv2D(filters = 64, kernel_size =(3,3), padding = 'valid',strides=(1,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),

    tf.keras.layers.Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid', strides=(1,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),                      
    tf.keras.layers.Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid', strides=(1,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid', strides=(1,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),


    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
    tf.keras.layers.Dense(2, activation='softmax')
     # model2 = load_model('../data/modelcheckpoint/myPproject_5.hdf5')
    ])
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5,epsilon=None), metrics=['acc'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-5,epsilon=None), metrics=['acc'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_acc', patience=40)
    lr = ReduceLROnPlateau(patience=5, factor=0.5,verbose=1)

    history = model.fit_generator(train_generator,epochs=50, steps_per_epoch= len(x_train) / 32,
    validation_data=valid_generator, callbacks=[early_stopping,lr])

    print(model.evaluate(valid_generator)[1])

    return model

   
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)


    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-2,epsilon=None), metrics=['acc'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_acc', patience=5)
    lr = ReduceLROnPlateau(patience=5, factor=0.5,verbose=1)

    history = model.fit_generator(train_generator,epochs=30, steps_per_epoch= len(x_train) / 32,
    validation_data=valid_generator, callbacks=[early_stopping,lr])

    print(model.evaluate(valid_generator)[1])

    return model

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")