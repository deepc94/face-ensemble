"""
Code to load the weights of pretrained VGG-Face Descriptor model and return
4096-D FC7 features.

Author: Deep Chakraborty
Date Created: 12/07/2017
Date Modified: 12/20/2017
"""
from __future__ import print_function
__author__ = "Deep Chakraborty"


import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.misc import imread, imsave, imresize
from scipy import spatial
import scipy.io as sio
# import cPickle as pickle

# import TensorflowUtils as utils
from lfw import *

def vgg_net (data, image):

	# read layer info
	layers = data['layers']
	current = image

	for layer in layers[0]:
		name = layer[0]['name'][0][0]

		# stop the forward pass after the fc7 layer
		if name == 'relu7':
			break

		# perform the appropriate layer operation
		layer_type = layer[0]['type'][0][0]
		if layer_type == 'conv':
			if name[:2] == 'fc':
				padding = 'VALID'
			else:
				padding = 'SAME'
			stride = layer[0]['stride'][0][0]
			kernel, bias = layer[0]['weights'][0][0]
			bias = np.squeeze(bias).reshape(-1)
			conv = tf.nn.conv2d(current, tf.constant(kernel),
								strides=(1, stride[0], stride[0], 1), padding=padding)
			current = tf.nn.bias_add(conv, bias)
			print(name, 'stride:', stride, 'kernel size:', np.shape(kernel))
		elif layer_type == 'relu':
			current = tf.nn.relu(current)
			print(name)
		elif layer_type == 'pool':
			stride = layer[0]['stride'][0][0]
			pool = layer[0]['pool'][0][0]
			current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
									 strides=(1, stride[0], stride[0], 1), padding='SAME')
			print(name, 'stride:', stride)
		elif layer_type == 'softmax':
			current = tf.nn.softmax(tf.reshape(current, [-1, 2622]))
			print(name)

	# return the fc7 values
	return current

def get_fc7 (image):
	"""
	Extract fc7 features from given image
	"""
	print("setting up vgg initialized conv layers ...")
	model_dir = './'
	model_name = 'vgg-face.mat'
	model_data = sio.loadmat(model_dir+model_name)
	fc7_layer = vgg_net(model_data, image)

	# return fc7_layer for the current batch of images
	return tf.reshape(fc7_layer, [-1, fc7_layer.get_shape().as_list()[3]])

def merge_fc7 (features, method):
	"""
	Merge fc7 features from different images using specified method

	Inputs: 
		features: array of feature vectors to be merged
		method: 'average': merge features by taking average
			'max_contrib': merge features by keeping maximal contributions

	Outputs:
		comb_feature: combined feature vector using 'method'
	"""
	# comb_feature = np.vstack(features) if features is a tuple

	if method == 'average':
		comb_feature = np.mean(features, axis=0)

	elif method == 'max_contrib':
		mask = np.argmax(np.absolute(features), axis=0)
		comb_feature = features[mask, np.arange(features.shape[1])]

	else: 
		raise Exception("Unexpected method: {}".format(method))

	return comb_feature

def similarity (feature1, feature2, method):
	"""
	Computes similarity between two fc7 feature vectors

	Inputs: 
		feature1: feature vector1
		feature2: feature vector2
		method: 'L2': compute similarity using L2 distance
			'cosine': compute similarity using cosine distance
			'rank1': computer similarity using rank1 score

	Outputs:
		score: similarity score between two feature vectors
	"""

	if method == 'L2':
		score = np.sqrt(np.sum((feature1-feature2)**2, axis=1))

	elif method == 'cosine':
		score = np.zeros(feature1.shape[0], dtype=np.float32)
		for i in range(feature1.shape[0]):
			score[i] = spatial.distance.cosine(feature1[i,:], feature2[i,:])

	elif method == 'rank1':
		pass

	else:
		raise Exception("Unexpected method: {}".format(method))

	return score


def main (argv=None):

	print("Start ...")
	image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_image")
	# Define placeholder for fc7 features of an image
	feature = get_fc7(image)

	print("Reading image pairs ...")

	# define image paths
	pairs_path = './dataset/pairsDev.txt'
	suffix = 'jpg'
	root = './dataset/lfw2'

	# determine image pairs to be loaded
	pairs = load_pairs(pairs_path, root, suffix)

	with tf.Session() as sess:

		# init tf session and get the feature vectors for the images
		print("Evaluating forward pass for VGG face Descriptor ...")
		sess.run(tf.global_variables_initializer())
		# define placeholders for 2 sets of images to be compared, as well as their labels
		feature1 = np.zeros([pairs.shape[0], 4096], dtype=np.float32)
		feature2 = np.zeros([pairs.shape[0], 4096], dtype=np.float32)

		# for combined features using average
		feature3 = np.zeros([pairs.shape[0], 4096], dtype=np.float32)
		feature4 = 	np.zeros([pairs.shape[0], 4096], dtype=np.float32)

		# for combined features using max_contib
		feature5 = np.zeros([pairs.shape[0], 4096], dtype=np.float32)
		feature6 = 	np.zeros([pairs.shape[0], 4096], dtype=np.float32)

		same = np.zeros([pairs.shape[0]], dtype=np.int32)

		# load the images
		i=0
		for pair in pairs:
			if i%10 == 0 or i==0:
				print("Evaluated {} pairs".format(i))
			name1, name2, same[i] = pairs_info(pair, suffix)
			name3, name4, _ = pairs_info_multiple(pair, suffix)

			image1, image2 = readImage(root, name1, name2)
			image3, image4 = readImage(root, name3, name4)

			feature1[i,:] = sess.run(feature, feed_dict={image: image1})
			feature2[i,:] = sess.run(feature, feed_dict={image: image2})

			f3 = sess.run(feature, feed_dict={image: image3})
			f4 = sess.run(feature, feed_dict={image: image4})

			feature3[i,:] = merge_fc7(f3, method='average')
			feature4[i,:] = merge_fc7(f4, method='average')
			feature5[i,:] = merge_fc7(f3, method='max_contrib')
			feature6[i,:] = merge_fc7(f4, method='max_contrib')
			i += 1

		print("Evaluated {} pairs".format(i))

	# find the similarity scores using different distance metrics
	dist_L2_normal = similarity(feature1, feature2, 'L2')
	dist_cos_normal = similarity(feature1, feature2, 'cosine')

	dist_L2_avg = similarity(feature3, feature4, 'L2')
	dist_cos_avg = similarity(feature3, feature4, 'cosine')

	dist_L2_max = similarity(feature5, feature6, 'L2')
	dist_cos_max = similarity(feature5, feature6, 'cosine')	

	# save the similarity scores for plotting histograms
	mat = np.vstack((dist_L2_normal, same)).T
	sio.savemat('dist_L2_normal.mat', {'mat':mat})
	mat = np.vstack((dist_cos_normal, same)).T
	sio.savemat('dist_cos_normal.mat', {'mat':mat})

	mat = np.vstack((dist_L2_avg, same)).T
	sio.savemat('dist_L2_avg.mat', {'mat':mat})
	mat = np.vstack((dist_cos_avg, same)).T
	sio.savemat('dist_cos_avg.mat', {'mat':mat})

	mat = np.vstack((dist_L2_max, same)).T
	sio.savemat('dist_L2_max.mat', {'mat':mat})
	mat = np.vstack((dist_cos_max, same)).T
	sio.savemat('dist_cos_max.mat', {'mat':mat})


if __name__ == "__main__":
    tf.app.run()
