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
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, Lambda, GlobalAveragePooling2D, Input
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
DATA = 'kings_slam.npy'
DATA_DP = 'kings_slam_dp.npy'
DATA_DT = 'kings_slam_dt.npy'

x_data = np.float32(np.load(DATA_DIR + DATA))
y_data_dp = np.float32(np.load(DATA_DIR + DATA_DP))
y_data_dt = np.float32(np.load(DATA_DIR + DATA_DT))

x1_data = x_data[0:-1]
x2_data = x_data[1:]

seq1 = 261
seq2 = 61
seq3 = 181
seq4 = 489
seq5 = 142
seq6 = 249
seq7 = 103
seq8 = 79 - 10 
seq_end = np.cumsum([0, seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8])

test_i = np.empty(shape=0)
i = 1
while i < len(seq_end):
    size = int(0.2*(seq_end[i] - seq_end[i-1]))
    test_inds = np.random.randint(seq_end[i-1], seq_end[i], size=size)
    test_i = np.append(test_i, test_inds)
    i += 1

set_test_i = set(test_i)
train_i = []
i = 0
while i < len(x1_data):
    if i not in set_test_i:
        train_i.append(i)
    i += 1
train_i = np.array(train_i, dtype=np.uint16)
test_i = np.array(test_i, dtype=np.uint16)

mean_x = np.mean(x1_data[train_i], axis=0)

x1_train = preprocess(x1_data[train_i], mean_x)
x2_train = preprocess(x2_data[train_i], mean_x)
x1_test = preprocess(x1_data[test_i], mean_x)
x2_test = preprocess(x2_data[test_i], mean_x)

y_train_x = y_data_dp[train_i]
y_train_q = y_data_dt[train_i]
y_test_x = y_data_dp[test_i]
y_test_q = y_data_dt[test_i]

print(str(len(x1_train)) + ' x1 Training samples, ' + str(len(x1_test)) + ' x1 Testing samples')
print(str(len(x2_train)) + ' x2 Training samples, ' + str(len(x2_test)) + ' x2 Testing samples')
print(str(len(y_train_x)) + ' y1 Training labels, ' + str(len(y_test_x)) + ' y1 Testing labels')
print(str(len(y_train_q)) + ' y2 Training labels, ' + str(len(y_test_q)) + ' y2 Testing labels')

### Parameters
img_width, img_height = 299, 299 
batch_size = 32
epochs1 = 50
epochs2 = 50
train_size = len(x1_train)
test_size = len(x1_test)


### Model
input_1 = Input(shape=(img_width, img_height, 3))
input_2 = Input(shape=(img_width, img_height, 3))
base_model = InceptionV3(weights='imagenet', input_shape=(img_width, img_height, 3), pooling=None, include_top=False)

branch_1 = base_model(input_1)
branch_2 = base_model(input_2)

x1 = GlobalAveragePooling2D()(branch_1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(rate=0.5)(x1)

x2 = GlobalAveragePooling2D()(branch_2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(rate=0.5)(x2)

x = concatenate([x1, x2])
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=0.5)(x)

output_positions = Dense(3, name='dx')(x)
output_quaternions = Dense(4, name='dq')(x)



def median(v):
  v = tf.reshape(v, [-1])
  m = batch_size//2
  return tf.nn.top_k(v, m).values[m-1]

def dx_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def dq_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred / tf.norm(y_pred, ord=2))

def median_dx(y_true, y_pred):
    return median(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1)))

def median_dq(y_true, y_pred):
    q1 = y_true
    q2 = K.l2_normalize(y_pred, axis=-1)
    d = tf.reduce_sum(tf.multiply(q1, q2), axis=-1)
    theta = 2 * tf.acos(d) * 180.0 / np.pi
    return median(theta)

model = Model(inputs=[input_1, input_2], outputs=[output_positions, output_quaternions])

for i, layer in enumerate(base_model.layers):
    layer.trainable = False


model.summary()
model.compile(
    optimizer=RMSprop(),
    loss={'dx': dx_loss, 'dq': dq_loss}, 
    loss_weights={'dx': 1, 'dq':10},
    metrics={'dx': median_dx, 'dq': median_dq})

history1 = fitData(batch_size, 
    epochs1, 
    model, 
    generatorSLAM(x1_train, x2_train, [y_train_x, y_train_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width)), 
    generatorSLAM(x1_test, x2_test, [y_test_x, y_test_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width)), 
    train_size, 
    test_size)

model.save_weights('kings_slam_top.h5')


for i, layer in enumerate(base_model.layers):
    layer.trainable = True

sgd = SGD(lr=1e-6, decay=0.99)
adam = Adam(lr=1e-5, clipvalue=1.5, decay=0.98)
model.compile(
    optimizer=adam,
    loss={'dx': dx_loss, 'dq': dq_loss}, 
    loss_weights={'dx': 1, 'dq':10},
    metrics={'dx': median_dx, 'dq': median_dq})

history2 = fitData(batch_size, 
    epochs2, 
    model, 
    generatorSLAM(x1_train, x2_train, [y_train_x, y_train_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width)), 
    generatorSLAM(x1_test, x2_test, [y_test_x, y_test_q], batch_size, preprocessing_function=None, target_dim=(img_height, img_width)), 
    train_size, 
    test_size)

model.save_weights('kings_slam_top.h5')


