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

DATA_DIR = 'KingsCollege/'
train = to_CSV(DATA_DIR + 'dataset_train.txt')
test = to_CSV(DATA_DIR + 'dataset_test.txt')

createSet(train, DATA_DIR + 'kings_train', 560, 315)
createSet(train, DATA_DIR + 'kings_test', 560, 315)