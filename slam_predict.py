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


x1_test = preprocess(x1_data, mean_x)
x2_test = preprocess(x2_data, mean_x)

y_test_x = y_data_dp
y_test_q = y_data_dt


### Parameters
img_width, img_height = 299, 299 
batch_size = 32
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


def median_dx(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))

def median_dq(y_true, y_pred):
    q1 = y_true
    q2 = K.l2_normalize(y_pred, axis=-1)
    d = tf.reduce_sum(tf.multiply(q1, q2), axis=-1)
    theta = 2 * tf.acos(d) * 180.0 / np.pi
    return theta

model = Model(inputs=[input_1, input_2], outputs=[output_positions, output_quaternions])
model.load_weights('weights/kings_slam_bottom.h5')

predictions = model.predict([x1_test, x2_test], batch_size=32, verbose=1)

yp = predictions[0]
yq = predictions[1]

metrics = np.zeros((len(yp), 2))

i = 0
while i < len(yp):
    metrics[i, 0] = np.linalg.norm(y_test_x[i] - yp[i])
    
    q1 = y_test_q[i] / np.linalg.norm(y_test_q[i])
    if np.isnan(np.min(q1)):
        q1 = 0
    q2 = yq[i] / np.linalg.norm(yq[i])
    if np.isnan(np.min(q2)):
        q2 = 0
    d = abs(np.sum(np.multiply(q1,q2)))
    theta = 2 * np.arccos(d) * 180/np.pi

    metrics[i, 1] = theta
    i += 1

median_result = np.median(metrics, axis=0)
print("Median error: " + str(median_result[0]) + 'm, ' + str(median_result[1]) + ' degrees')

np.save('predictions/position_king_slam', yp)
np.save('predictions/orientation_king_slam', yq)
np.save('predictions/orientation_king_slam_metrics', metrics)
np.save('predictions/y_labels_test_x', y_test_x)