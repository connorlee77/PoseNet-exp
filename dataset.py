import os
import shutil

import numpy as np 
import pandas as pd 

import cv2 


def to_CSV(filepath):
	return pd.read_csv(filepath, delim_whitespace=True, header=None, names=['path', 'X', 'Y', 'Z', 'W', 'P', 'Q', 'R'], skiprows=3)
	



def createSet(data, filename, width, height):
	x = np.empty((len(data), height, width, 3))

	for index, row in data.iterrows():
		path = row['path']

		pic = cv2.imread(DATA_DIR + path)

		h = pic.shape[0]
		w = pic.shape[1]

		factor = height / float(min(h, w)) 
		pic = cv2.resize(pic, (int(w * factor), int(h * factor)))

		x[index] = pic

	x = np.uint8(x)
	np.save(filename, x)
	y = data.as_matrix(['X', 'Y', 'Z', 'W', 'P', 'Q', 'R'])
	assert len(x) == len(y)
	np.save(filename + '_y', y)

# DATA_DIR = 'KingsCollege/'
# TRAIN_DST = 'kings_train'
# TEST_DST = 'kings_test'

# DATA_DIR = 'OldHospital/'
# TRAIN_DST = 'hospital_train'
# TEST_DST = 'hospital_test'

DATA_DIR = 'GreatCourt/'
TRAIN_DST = 'court_train'
TEST_DST = 'court_test'

train = to_CSV(DATA_DIR + 'dataset_train.txt')
test = to_CSV(DATA_DIR + 'dataset_test.txt')

createSet(train, DATA_DIR + TRAIN_DST, 560, 315)
createSet(test, DATA_DIR + TEST_DST, 560, 315)