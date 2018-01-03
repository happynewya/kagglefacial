#coding=utf-8

import tensorflow as tf 
from trainer import *
from datamanager import *
from net import CNN
import matplotlib.pyplot as plt


def plotimg(point, x_value):
	image = x_value.reshape(96, 96)
	plt.imshow(image, cmap='gray')
	plt.scatter(point[0::2]*48 +48, point[1::2]*48+48,marker='x',s=10)
	plt.show()


def train(**kwargs):
	model = kwargs.get('model', 1)
	if model == 1:
		outputsize = 8
		tfpath = './data/train1'
	elif model == 2:
		outputsize = 22
		tfpath = './data/train2'
	cnn = CNN(outputsize = outputsize)
	with tf.Graph().as_default():
		dmgr = DataManager(dir = tfpath, model = model, count = 2000, batchsize = 200)
		trainer = Solver(data=dmgr, net=cnn, model = model, maxiter = 2000, lr = 0.001)
		print("Start training model ", model," : ")
		trainer.train()
		print("Complete training model ", model)

def predict(**kwargs):
	model = kwargs.get('model', 1)
	if model == 1:
		outputsize = 8
	elif model == 2:
		outputsize = 22
	cnn = CNN(outputsize = outputsize)
	with tf.Graph().as_default():
		dmgr= DataManager(dir = './data/test.tfrecords', batchsize = 1, test=True, model=model)
		evaler = Solver(data=dmgr, net=cnn, batchsize = 1, test = True, model = model)
		y, x = evaler.predict()
	return y, x




if __name__=='__main__':
	#train(model = 1)
	y,x = predict(model = 1)
