
import tensorflow as tf 



def parsefun1(example_proto):
	features = {
		"image": tf.FixedLenFeature((96*96), tf.float32),
		"label": tf.FixedLenFeature((8), tf.float32, default_value = [0.0] * 8)
	}
	pf = tf.parse_single_example(example_proto, features)
	img = tf.reshape(pf["image"], (96, 96, -1))
	return img, pf["label"]

def parsefun2(example_proto):
	features = {
		"image": tf.FixedLenFeature((96*96), tf.float32),
		"label": tf.FixedLenFeature((22), tf.float32, default_value = [0.0] * 22)
	}
	pf = tf.parse_single_example(example_proto, features)
	img = tf.reshape(pf["image"], (96, 96, -1))
	return img, pf["label"]

#split is size
# model1:
#  train: 6300
#	validation: 700 
#model2:
#  train:1508
#  validation: 647
#

class DataManager:
	def __init__(self, **kwargs):
		self.path = kwargs.get('dir')
		self.test = kwargs.get('test', False)
		self.model = kwargs.get('model', 1)
		self.batchsize = kwargs.get('batchsize', 50)
		if self.test:
			self.batchsize = 1
		self.count = kwargs.get('count', None)
		if self.test:
			self.count = 1
		self.shuf_buf_size = kwargs.get('shuf_buf_size', None)
		if self.model == 1:
			self.parsefun = parsefun1
			self.vsize = 700 
		elif self.model == 2:
			self.parsefun = parsefun2
			self.vsize = 647
		if not self.test:
			self.dataset = tf.data.TFRecordDataset(self.path+ '_train.tfrecords')
		else:
			self.dataset = tf.data.TFRecordDataset(self.path)
		self.dataset = self.dataset.map(self.parsefun)
		self.dataset = self.dataset.batch(self.batchsize)
		self.dataset = self.dataset.repeat(self.count)
		if not self.test:
			if self.shuf_buf_size is not None:
				self.dataset = self.dataset.shuffle(buffer_size = self.shuf_buf_size)
			self.vdataset = tf.data.TFRecordDataset(self.path + '_test.tfrecords')#for validation
			self.vdataset = self.vdataset.map(self.parsefun)
			self.vdataset = self.vdataset.batch(self.vsize)
			self.vdataset = self.vdataset.repeat(None)

	def next_batch(self):
		with tf.name_scope('input'):
			iterator = self.dataset.make_one_shot_iterator()
			batch = iterator.get_next()
		return batch
	def get_vbatch(self):
		iterator = self.vdataset.make_one_shot_iterator()	
		batch = iterator.get_next()
		return batch
	def getbatchsize(self):
		return self.batchsize
