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
DATA_DIR = 'OldHospital/'
TEST = 'hospital_test.npy'
TEST_Y = 'hospital_test_y.npy'

### Data gathering & preprocessing
x_test = np.float32(np.load(DATA_DIR + TEST))
y_test = np.float32(np.load(DATA_DIR + TEST_Y))

y_test_x = y_test[:,0:3]
y_test_q = y_test[:,3:]

x_test = preprocess(x_test)

test_set = np.zeros((len(x_test), 299, 299, 3))
i = 0
while i < len(x_test):
    test_set[i] = centerCrop(x_test[i])
    i += 1

print(str(len(test_set)) + ' Testing samples')

### Parameters
img_width, img_height = 299, 299 
batch_size = 32
train_size = 1200
test_size = len(x_test)

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
model.load_weights('weights/hospital_bottom.h5')

predictions = model.predict(test_set, batch_size=32, verbose=1)

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
print "Median error: " + str(median_result[0]) + 'm, ' + str(median_result[1]) + ' degrees'



