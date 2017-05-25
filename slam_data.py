import os
import shutil

import numpy as np 
import pandas as pd 

import cv2 
import progressbar
from pyquaternion import Quaternion

def to_CSV(filepath):
	return pd.read_csv(filepath, delim_whitespace=True, header=None, names=['path', 'X', 'Y', 'Z', 'W', 'P', 'Q', 'R'], skiprows=3)

DATA_DIR = 'KingsCollege/'
TRAIN_DST = 'kings_train'
TEST_DST = 'kings_test'

train = to_CSV(DATA_DIR + 'dataset_train.txt')
test = to_CSV(DATA_DIR + 'dataset_test.txt')

def merge(train, test):
	df = pd.concat([train, test])
	return df.sort(columns=['path'])

def centerCrop(x, target_dim=(299, 299), currImgDim=(315, 560)):
    img_height, img_width = target_dim
    currHeight, currWidth = currImgDim

    x = np.squeeze(x)
    sX = (currWidth // 2) - (img_width//2) 
    sY = (currHeight // 2) - (img_height//2)
    return x[sY:sY + img_height, sX:sX + img_width] 


def odom_data(train, test, width, height, filename):
	data = merge(train, test)

	x = np.zeros((len(data), 299, 299, 3), dtype=np.uint8)
	
	dp = np.zeros((len(data) - 1, 3))
	dt = np.zeros((len(data) - 1, 4))

	bar = progressbar.ProgressBar()
	for i in bar(range(1, len(data))):
		prev_row = data.iloc[i - 1]
		curr_row = data.iloc[i]

		prev_path = prev_row['path']
		curr_path = curr_row['path']

		if os.path.dirname(prev_path) != os.path.dirname(curr_path):
			continue

		prev_pic = cv2.imread(DATA_DIR + prev_path)
		curr_pic = cv2.imread(DATA_DIR + curr_path)
		h = prev_pic.shape[0]
		w = prev_pic.shape[1]

		factor = height / float(min(h, w)) 
		prev_pic = cv2.resize(prev_pic, (int(w * factor), int(h * factor)))
		curr_pic = cv2.resize(curr_pic, (int(w * factor), int(h * factor)))


		prev_pic = centerCrop(prev_pic, target_dim=(299, 299), currImgDim=(315, 560))
		curr_pic = centerCrop(curr_pic, target_dim=(299, 299), currImgDim=(315, 560))

		if i == 1:
			x[i - 1] = prev_pic
		x[i] = curr_pic

		px, py, pz, pa, pb, pc, pd  = prev_row.as_matrix(['X', 'Y', 'Z', 'W', 'P', 'Q', 'R'])
		cx, cy, cz, ca, cb, cc, cd  = curr_row.as_matrix(['X', 'Y', 'Z', 'W', 'P', 'Q', 'R'])

		pq = Quaternion(pa, pb, pc, pd)
		cq = Quaternion(ca, cb, cc, cd)
		
		rq = cq / pq 
		assert len(rq.elements) == 4 

		dp[i - 1] = np.array([cx - px, cy - py, cz - pz])
		dt[i - 1] = rq.elements

	np.save(filename + '_slam_dt', dt)
	np.save(filename + '_slam_dp', dp)
	np.save(filename + '_slam', x)

odom_data(train, test, 560, 315, DATA_DIR + 'kings')
