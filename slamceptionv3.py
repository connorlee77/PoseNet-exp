import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(148)

import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, Lambda, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam

from generator import generatorSLAM

def fitData(batch_size, epochs, model, generator_train, generator_test, train_size, test_size):
    tbCB = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)

    history = model.fit_generator(generator_train,
        steps_per_epoch=train_size / batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=test_size / batch_size,
        callbacks=[tbCB])

    return history

def preprocess(x, mean_x):
    x = (x - mean_x) / 255.0
    return x

### File paths
DATA_DIR = 'KingsCollege/'
data = None
data_dp = None
data_dx = None

### TODO: Data gathering & preprocessing: update for slam
x_train = None
x_test = None

y_train = None
y_test = None

y_train_x = None
y_train_q = None
y_test_x = None
y_test_q = None

mean_x = np.mean(x_train, axis=0)
x_train = preprocess(x_train, mean_x)
x_test = preprocess(x_test, mean_x)

print(str(len(x_train)) + ' Training samples, ' + str(len(x_test)) + ' Testing samples')

### Parameters
img_width, img_height = 299, 299 
batch_size = 32
epochs1 = 50
epochs2 = 50
train_size = len(x_train)
test_size = len(x_test)



### Model

base_model1 = InceptionV3(weights='imagenet', input_shape=(img_width, img_height, 3), pooling=None, include_top=False)
base_model2 = InceptionV3(weights='imagenet', input_shape=(img_width, img_height, 3), pooling=None, include_top=False)


# Top classifiers


def median(v):
  v = tf.reshape(v, [-1])
  m = batch_size//2
  return tf.nn.top_k(v, m).values[m-1]

def x_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def q_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred, ord=2)

# TODO: Combined model w/ classifier






### TODO: update generator Training 

# history1 = fitData(batch_size, 
#     epochs1, 
#     model, 
#     generatorSLAM(x_train, [y_train_x, y_train_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width), currImgDim=(currHeight, currWidth)), 
#     generatorSLAM(x_test, [y_test_x, y_test_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width), currImgDim=(currHeight, currWidth)), 
#     train_size, 
#     test_size)

# model.save_weights('kings_slam_top.h5')


# for i, layer in enumerate(base_model.layers):
#     layer.trainable = True

# sgd = SGD(lr=1e-6, decay=0.99)
# adam = Adam(lr=1e-5, clipvalue=1.5)
# model.compile(
#     optimizer=adam,
#     loss={'x': x_loss, 'q': q_loss}, 
#     loss_weights={'x': 1, 'q':350},
#     metrics={'x': median_x, 'q': median_q})

# history2 = fitData(batch_size, 
#     epochs2, 
#     model, 
#     generatorSLAM(x_train, [y_train_x, y_train_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width), currImgDim=(currHeight, currWidth)), 
#     generatorSLAM(x_test, [y_test_x, y_test_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width), currImgDim=(currHeight, currWidth)), 
#     train_size, 
#     test_size)


# model.save_weights('kings_slam_top.h5')


