#coding=utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import time, sys, os
import net
from net import *
import numpy as np
import pandas as pd

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


def plotimg(point, x_value):
	image = x_value.reshape(96, 96)
	plt.imshow(image, cmap='gray')
	plt.scatter(point[0::2] * 48 + 48, point[1::2]*48+48,marker='x',s=10)
	plt.show()



class Solver(object):
	def __init__(self, **kwargs):
		self.maxiter = int(kwargs.get('maxiter', 500))
		self.lr = float(kwargs.get('lr', 0.02))

		self.snapshot_dir = kwargs.get('snapshot_dir', './model')
		self.snapshot_iter = int(kwargs.get('snapshot_dir', 200))

		self.summary_iter = int(kwargs.get('summary_iter', 50))
		self.summary_dir = kwargs.get('summary_dir', './summary')

		self.model = kwargs.get('model', 1)
		if self.model == 1:
			self.tsize = 4900
			self.outsize = 8
		elif self.model == 2:
			self.tsize = 1508
			self.outsize = 22

		self.test = kwargs.get('test', False)
		self.data = kwargs.get('data', None)
		self.net = kwargs.get('net', None)


	def train(self):
		x, labels = self.data.next_batch()
		xv, lv = self.data.get_vbatch()

		x_ = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])
		y_ = tf.placeholder(tf.float32, shape=[None, self.outsize])
		y = self.net.inference(x_)

		J_loss = self.net.loss(y_, y)
		eval_loss = self.net.eval(y_, y)

		J_scalar = tf.summary.scalar('loss', J_loss)
		eval_scalar = tf.summary.scalar('eval', eval_loss)
		

		with tf.variable_scope('optimizer'):
			self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(J_loss)

		init_op = tf.global_variables_initializer()
		summary_op = tf.summary.merge([J_scalar])
		summary_op_eval = tf.summary.merge([eval_scalar])
		summary_writer = tf.summary.FileWriter(
			os.path.join(self.summary_dir, 'train{}'.format(self.model)))
		summary_writer.add_graph(tf.get_default_graph())

		saver = tf.train.Saver()
		checkpointloc = './model/model' + str(self.model)
		checkpoint =tf.train.latest_checkpoint(checkpointloc)
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.3
		with tf.Session(config=config) as sess:
			sess.run(init_op)
			if checkpoint is not None: 
				saver.restore(sess, checkpoint)
				print('restored variables from {}'.format(checkpoint))
			for epoch in range(1, self.maxiter + 1):
				try:
					for _ in range(int(self.tsize / self.data.getbatchsize())):
						start_time = time.time()
						x_v, labels_v = sess.run([x, labels])
						self.train_op.run(feed_dict={x_: x_v, y_:labels_v})
						end_time = time.time()
						dur = end_time - start_time
				except tf.errors.OutOfRangeError:
					break
				summary, loss= sess.run([summary_op, J_loss], feed_dict={x_:x_v, y_:labels_v})
				summary_writer.add_summary(summary, epoch)
				xv_, lv_ = sess.run([xv, lv])
				summary, e_loss= sess.run([summary_op_eval, eval_loss], feed_dict={x_:xv_, y_:lv_})
				summary_writer.add_summary(summary, epoch)
				
				epsec = self.data.batchsize / dur
				print(('epoch %6d: eval=%.4f loss = %.4f(%.1f example/sec)') % (epoch, e_loss, loss, epsec))
				sys.stdout.flush()

				if ( epoch == self.maxiter ) or (epoch % self.snapshot_iter == 0):
					saver.save(sess, './model/model'+str(self.model)+'/{}model.ckpt'.format(self.model), global_step = epoch)
					print("model saved.")

	def predict(self):
		np.set_printoptions(precision=5)
		x, label = self.data.next_batch()
		y = self.net.inference(x)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			for i in range(1783):
				checkpointloc = './model/model' + str(self.model)
				checkpoint =tf.train.latest_checkpoint(checkpointloc)
				saver.restore(sess, checkpoint)
				if i == 0:
					result = sess.run(y)				
					result = result * 48 + 48
					result = np.round(result, decimals=10)
				else:
					tmp = sess.run(y)				
					tmp = tmp * 48 + 48
					tmp = np.round(tmp, decimals=10)
					result = np.concatenate((result, tmp))
		if self.model == 1:
			df = pd.DataFrame(data = result, columns = FEATURE1)
		elif self.model == 2:
			df = pd.DataFrame(data = result, columns = FEATURE2)
		df.to_csv("./result/result" + str(self.model) +".csv",  sep=",", index=False)
		return 2,2

