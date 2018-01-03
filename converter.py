

import tensorflow as tf
import pandas as pd 
from pandas.io.parsers import read_csv
import os
import numpy as np

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

FEATURE1=[
'left_eye_center_x'  ,
'left_eye_center_y'  ,
'right_eye_center_x' ,
'right_eye_center_y' ,
'nose_tip_x',
'nose_tip_y',
'mouth_center_bottom_lip_x',
'mouth_center_bottom_lip_y'
]

FEATURE2 = [
'left_eye_inner_corner_x'  ,
'left_eye_inner_corner_y'  ,
'left_eye_outer_corner_x'  ,
'left_eye_outer_corner_y'  ,
'right_eye_inner_corner_x' ,
'right_eye_inner_corner_y' ,
'right_eye_outer_corner_x' ,
'right_eye_outer_corner_y' ,
'left_eyebrow_inner_end_x' ,
'left_eyebrow_inner_end_y' ,
'left_eyebrow_outer_end_x' ,
'left_eyebrow_outer_end_y' ,
'right_eyebrow_inner_end_x',
'right_eyebrow_inner_end_y',
'right_eyebrow_outer_end_x',
'right_eyebrow_outer_end_y',
'mouth_left_corner_x'    ,
'mouth_left_corner_y'    ,
'mouth_right_corner_x'   ,
'mouth_right_corner_y'   ,
'mouth_center_top_lip_x' ,
'mouth_center_top_lip_y' 
]


SPLIT = 0.9

def loadcsv(test = False, cols = None):
	fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols:
		df = df[list(cols)+['Image']]
	df = df.dropna()
	print(df.count())
	images = np.vstack(df['Image'].values) / 255
	images = images.astype(np.float32)

	if fname == FTRAIN:
		label = df[df.columns[:-1]].values
		label = (label - 48)/48
		label = label.astype(np.float32)
	else:
		label = None
	images = images.reshape(-1, 96, 96, 1)
	return images, label


def savetfrecord(path, images, labels, split = True):
	testpath = ''
	trainpath = "{}.tfrecords".format(path)
	if not path.endswith('.tfrecords') and split:
		testpath = "{}_test.tfrecords".format(path)
		trainpath = "{}_train.tfrecords".format(path)
	print("saving tfrecord：", path)
	tfwriter_train = tf.python_io.TFRecordWriter(trainpath)
	if split:
		tfwriter_test = tf.python_io.TFRecordWriter(testpath)
	TRAINSIZE = int(images.shape[0] * SPLIT);
	for i in range(images.shape[0]):
		feature = {}
		if labels is not None:
			feature['label'] = tf.train.Feature(float_list = tf.train.FloatList(value = labels[i]))
		feature['image'] = tf.train.Feature(
			float_list = tf.train.FloatList(value = np.ravel(images[i])))
		example = tf.train.Example(features = tf.train.Features(feature = feature))
		if split:
			if i < TRAINSIZE:
				tfwriter_train.write(example.SerializeToString())
			else:
				tfwriter_test.write(example.SerializeToString())
		else:
			tfwriter_train.write(example.SerializeToString())
	print("finised.")

if __name__ == '__main__':
	images1, labels1 = loadcsv(False, FEATURE1)
	savetfrecord('./data/train1', images1, labels1)

	#images2, labels2 = loadcsv(False, FEATURE2)
	#savetfrecord('./data/train2', images2, labels2)

	#images3, labels3 = loadcsv(True)
	#savetfrecord('./data/test', images3, labels3, False)
