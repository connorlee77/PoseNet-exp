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

from generator import generator
from generator import randomCrop
from generator import centerCrop

def fitData(batch_size, epochs, model, generator_train, generator_test, train_size, test_size):
    tbCB = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)

    history = model.fit_generator(generator_train,
        steps_per_epoch=train_size / batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=test_size / batch_size,
        callbacks=[tbCB])

    return history

def preprocess(x):
    mean_x = np.mean(x, axis=0)
    x = (x - mean_x) / 255.0
    return x

### File paths
DATA_DIR = 'KingsCollege/'
TEST = 'kings_test.npy'
TEST_Y = 'kings_test_y.npy'

### Data gathering & preprocessing
x_test = np.float32(np.load(DATA_DIR + TEST))
y_test = np.float32(np.load(DATA_DIR + TEST_Y))

y_test_x = y_test[:,0:3]
y_test_q = y_test[:,3:]

x_test = preprocess(x_test)

print(str(len(x_test)) + ' Testing samples')

### Parameters
img_width, img_height = 299, 299 
batch_size = 32
train_size = 1200
test_size = 340

### Cropping functions
currHeight = 315
currWidth = 560

### Model
base_model = InceptionV3(weights='imagenet', input_shape=(img_width, img_height, 3), pooling=None, include_top=False)

# Top classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)

output_positions = Dense(3, name='x')(x)
output_quaternions = Dense(4, name='q')(x)

# Combined model w/ classifier
model = Model(inputs=base_model.input, outputs=[output_positions, output_quaternions])
model.summary()
model.load_weights('kings_bottom.h5')

def median(v):
  v = tf.reshape(v, [-1])
  m = batch_size//2
  return tf.nn.top_k(v, m).values[m-1]


def x_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def q_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred / tf.norm(y_pred, ord=2))

def median_x(y_true, y_pred):
    return median(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1)))

def median_q(y_true, y_pred):
    q1 = y_true
    q2 = K.l2_normalize(y_pred, axis=-1)
    d = tf.reduce_sum(tf.multiply(q1, q2), axis=-1)
    theta = 2 * tf.acos(d) * 180.0 / np.pi
    return median(theta)

model.compile(
    optimizer=RMSprop(),
    loss={'x': x_loss, 'q': q_loss}, 
    loss_weights={'x': 1, 'q':350},
    metrics={'x': median_x, 'q': median_q})

test_gen = generator(x_test, 
    [y_test_x, y_test_q], 
    batch_size, 
    preprocessing_function=centerCrop, 
    target_dim=(img_height, img_width), 
    currImgDim=(currHeight, currWidth))

print('testing')
print(model.evaluate_generator(test_gen, test_size))



