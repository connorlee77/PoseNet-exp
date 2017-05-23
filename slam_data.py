import os
import shutil

import numpy as np 
import pandas as pd 

import cv2 
import progressbar


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

def quat2transform(x, y, z, w):

	xx2 = 2 * x * x
	yy2 = 2 * y * y
	zz2 = 2 * z * z
	xy2 = 2 * x * y
	wz2 = 2 * w * z
	zx2 = 2 * z * x
	wy2 = 2 * w * y
	yz2 = 2 * y * z
	wx2 = 2 * w * x

	rmat = np.empty((3, 3), float)
	rmat[0,0] = 1. - yy2 - zz2
	rmat[0,1] = xy2 - wz2
	rmat[0,2] = zx2 + wy2
	rmat[1,0] = xy2 + wz2
	rmat[1,1] = 1. - xx2 - zz2
	rmat[1,2] = yz2 - wx2
	rmat[2,0] = zx2 - wy2
	rmat[2,1] = yz2 + wx2
	rmat[2,2] = 1. - xx2 - yy2

	return rmat

def odom_data(train, test, width, height, filename):
	data = merge(train, test)

	x = np.empty((len(data), 299, 299, 3), dtype=np.uint8)
	
	dp = np.empty((len(data) - 1, 3))
	dt = np.empty((len(data) - 1, 3, 3))

	bar = progressbar.ProgressBar()
	for i in bar(range(1, len(data))):
		prev_row = data.iloc[i - 1]
		curr_row = data.iloc[i]

		prev_path = prev_row['path']
		curr_path = curr_row['path']

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

		p_rot = quat2transform(pa, pb, pc, pd)
		c_rot = quat2transform(ca, cb, cc, cd)

		transformation_matrix = np.linalg.inv(p_rot).dot(c_rot)

		dp[i - 1] = np.array([cx - px, cy - py, cz - pz])
		dt[i - 1] = transformation_matrix

	np.save(filename + '_slam_dt', dt)
	np.save(filename + '_slam_dp', dp)
	np.save(filename + '_slam', x)

odom_data(train, test, 560, 315, DATA_DIR + 'kings')
