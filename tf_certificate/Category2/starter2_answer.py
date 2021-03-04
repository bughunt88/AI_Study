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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # (60000, 28, 28) (60000,) (10000, 28, 28)
    x_train = x_train / 255.
    x_test = x_test / 255.

    model = Sequential()
    model.add(Conv1D(256, 3, padding= 'valid', activation= 'relu', input_shape = (28,28)))
    model.add(Conv1D(128, 3, padding= 'valid', activation= 'relu'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    es = EarlyStopping(patience = 10)
    lr = ReduceLROnPlateau(factor = 0.25, patience = 5, verbose = 1)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = 'acc')
    model.fit(x_train, y_train, validation_split = 0.2, epochs = 1000, callbacks = [es, lr])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("../tf_certificate/Category2/mymodel.h5")

