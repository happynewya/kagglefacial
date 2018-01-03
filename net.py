#coding=utf-8

"""
	mother fucker Conv network for face alignment 
	15 keypoints(kaggle dataset)
"""

import tensorflow as tf

class CNN(object):

	def __init__(self, **kwargs):
		self.outputsize = kwargs.get('outputsize')
		assert self.outputsize == 8 or self.outputsize == 22
	def __conv(self, x, **kwargs):
		inputsize = x.get_shape()[-1]
		#print(inputsize)
		kernelsize = kwargs.get('kernelsize')
		stride = kwargs.get('stride', 1)
		kernelnum = kwargs.get('kernelnum')
		weightdecay = kwargs.get('wd', 0)
		weight = tf.get_variable(
			name = 'weight', 
			shape = [kernelsize, kernelsize, inputsize, kernelnum],
			initializer = tf.contrib.layers.xavier_initializer()
			)
		bias = tf.get_variable(
			name = 'bias',
			shape = [kernelnum],
			initializer = tf.constant_initializer(0.0)
			)
		if weightdecay != 0:
			wd_loss = tf.multiply(tf.nn.l2_loss(weight), weightdecay, name = 'wd_loss')
			tf.add_to_collection('losses', wd_loss)
		hconv = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME')
		h = tf.nn.relu(hconv, name = tf.get_default_graph().get_name_scope())
		return h

	def __pool(self, x, **kwargs):
		size = kwargs.get('size', 2)
		h = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides = [1, size, size, 1], padding = 'SAME')
		return h

	def __fc(self, x, **kwargs):
		inputsize = x.get_shape()[-1].value
		size = kwargs.get('size')
		weightdecay = kwargs.get('wd', 0)
		relu = kwargs.get('relu', True)
		weight = tf.get_variable(
			name = 'weight',
			shape = [inputsize, size],
			initializer = tf.contrib.layers.xavier_initializer()
			)
		bias = tf.get_variable(
			name = 'bias',
			shape = [size],
			initializer = tf.constant_initializer(0.0)
			)
		if weightdecay != 0:
			wd_loss = tf.multiply(tf.nn.l2_loss(weight), weightdecay, name = "wd_loss")
			tf.add_to_collection("losses", wd_loss)

		hfc = tf.matmul(x, weight) + bias
		if relu:
			h = tf.nn.relu(hfc, name = tf.get_default_graph().get_name_scope())
		else:
			h = hfc
		return h

	def inference(self, x):
		with tf.variable_scope('conv1'):
			h_conv1 = self.__conv(x, kernelsize = 3, stride = 1, kernelnum = 32)
			h_pool1 = self.__pool(h_conv1, size = 2)
		with tf.variable_scope('conv2'):
			h_conv2 = self.__conv(h_pool1, kernelsize = 2, stride = 1, kernelnum = 64)
			h_pool2 = self.__pool(h_conv2, size = 2)
		with tf.variable_scope('conv3'):
			h_conv3 = self.__conv(h_pool2, kernelsize = 2, stride = 1, kernelnum = 128)
			h_pool3 = self.__pool(h_conv3, size = 2)

		hshape = h_pool3.get_shape().as_list()
		#print("hshape: ", hshape)
		h_flattened = tf.reshape(h_pool3, [-1, hshape[1]*hshape[2]*hshape[3]])
		with tf.variable_scope('fc1'):
			h_fc1 = self.__fc(h_flattened, size = 500)
		with tf.variable_scope('fc2'):
			h_fc2 = self.__fc(h_fc1, size = 500)
		with tf.variable_scope('output'):
			h_output = self.__fc(h_fc2, size = self.outputsize, relu = False)
		y = h_output
		return y
	def eval(self, y, label):
		with tf.variable_scope('eval'):
			loss = tf.reduce_mean(tf.square(y-label), name = 'mse')
		return loss

	def loss(self, y, label):
		with tf.variable_scope('losses'):
			loss = tf.reduce_mean(tf.square(y-label), name = 'mse')
		return loss
