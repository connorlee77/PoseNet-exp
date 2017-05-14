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
from keras.optimizers import SGD, RMSprop

from PIL import Image

def fitData(batch_size, epochs, model, generator_train, generator_test, train_size, test_size):
    tbCB = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)

    history = model.fit_generator(generator_train,
        steps_per_epoch=train_size / batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=test_size / batch_size,
        callbacks=[tbCB])

    return history


### File paths
DATA_DIR = 'KingsCollege/'
TRAIN = 'kings_train.npy'
TEST = 'kings_test.npy'
TRAIN_Y = 'kings_train_y.npy'
TEST_Y = 'kings_test_y.npy'

### Data gathering & preprocessing
x_train = np.float32(np.load(DATA_DIR + TRAIN))
x_test = np.float32(np.load(DATA_DIR + TEST))


y_train = np.float32(np.load(DATA_DIR + TRAIN_Y))
y_test = np.float32(np.load(DATA_DIR + TEST_Y))

y_train_x = y_train[:,0:3]
y_train_q = y_train[:,3:]
y_test_x = y_test[:,0:3]
y_test_q = y_test[:,3:]

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

print(str(len(x_train)) + ' Training samples, ' + str(len(x_test)) + ' Testing samples')

### Parameters
img_width, img_height = 299, 299 
batch_size = 32
epochs1 = 15
epochs2 = 10
# train_size = len(x_train)
# test_size = len(x_test)
train_size = 1200
test_size = 1200
BETA = 500

### Cropping functions
currHeight = 315
currWidth = 560


def randomCrop(x):
    x = np.squeeze(x)
    rY = np.random.randint(0, currHeight - img_height)
    rX = np.random.randint(0, currWidth - img_width)
    return x[rY:rY + img_height, rX:rX + img_width]

def centerCrop(x):
    x = np.squeeze(x)
    sX = (currWidth // 2) - (img_width//2) 
    sY = (currHeight // 2) - (img_height//2)
    return x[sY:sY + img_height, sX:sX + img_width] 

def generator(features, labels, batch_size, preprocessing_function=None):

    q = labels[1]
    x = labels[0]

    batch_features = np.zeros((batch_size, img_height, img_width, 3))
    batch_x = np.zeros((batch_size, 3))
    batch_q = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            index = np.random.choice(len(features),1)

            batch_features[i] = preprocessing_function(features[index])

            batch_x[i] = x[index]
            batch_q[i] = q[index]

        yield batch_features, {'x': batch_x, 'q': batch_q}

# def directory_generator(train_dist_filepath, batch_size, preprocessing_function=None):

#     data = pd.read_csv(train_dist_filepath, delim_whitespace=True, header=None, names=['path', 'X', 'Y', 'Z', 'W', 'P', 'Q', 'R'], skiprows=3)

#     batch_features = np.zeros((batch_size, img_height, img_width, 3))
#     batch_x = np.zeros((batch_size, 3))
#     batch_q = np.zeros((batch_size, 4))

#     while True:
#         for i in range(batch_size):
#             index = np.random.choice(len(data), 1)

#             fpath = data.iloc[index]['path'].values[0]
#             f = Image.open(DATA_DIR + fpath)

#             f = f.resize((int(f.width * 315 / f.height), int(f.height * 315 / f.height)))

#             f = np.array(f)
#             if preprocessing_function:
#                 f = preprocessing_function(f)
#             f = preprocess_input(np.float32(f))
#             batch_features[i] = f

#             batch_x[i] = np.float32(data.iloc[index].as_matrix(['X', 'Y', 'Z']))
#             batch_q[i] = np.float32(data.iloc[index].as_matrix(['W', 'P', 'Q', 'R']))

#         yield batch_features, {'x': batch_x, 'q': batch_q}


### Model

base_model = InceptionV3(weights='imagenet', input_shape=(img_width, img_height, 3), pooling=None, include_top=False)

# Top classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)

output_positions = Dense(3, name='x')(x)
output_quaternions = Dense(4, name='q')(x)

def x_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def q_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred / tf.norm(y_pred, ord=2))

# Combined model w/ classifier
model = Model(inputs=base_model.input, outputs=[output_positions, output_quaternions])

model.summary()
model.compile(optimizer=SGD(lr=1e-5, decay=0.99, momentum=0.9), loss={'x': x_loss, 'q': q_loss}, loss_weights=[1, 500])

history1 = fitData(batch_size, 
    epochs1, 
    model, 
    generator(x_train, [y_train_x, y_train_q], batch_size, preprocessing_function=randomCrop), 
    generator(x_test, [y_test_x, y_test_q], batch_size, preprocessing_function=centerCrop), 
    train_size, 
    test_size)

# history1 = fitData(batch_size, 
#     epochs1, 
#     model, 
#     directory_generator(DATA_DIR + 'dataset_train.txt', batch_size, preprocessing_function=randomCrop), 
#     directory_generator(DATA_DIR + 'dataset_test.txt', batch_size, preprocessing_function=randomCrop), 
#     train_size, 
#     test_size)

model.save_weights('kings_top.h5')


