"""
Code to load the weights of pretrained VGG-Face Descriptor model and return
4096-D FC7 features.

Author: Deep Chakraborty
Date Created: 12/07/2017
Date Modified: 12/18/2017
"""

__author__ = "Deep Chakraborty"

import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.misc import imread, imsave, imresize
import scipy.io as sio
import cPickle as pickle

import TensorflowUtils as utils
from lfw import *

def vgg_net (weights, image):

	"""
	VGG Architecture:

	layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
		'relu3_3', 'pool3',

		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
		'relu4_3', 'pool4',

		'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
		'relu5_3', 'pool5',

		'fc6', 'relu6', 'dropout6',
		'fc7', 'relu7', 
		'fc8', 'prob'
	)
	"""
	layer_num = weights.shape[0]
	net = {}
	current = image

	for i in xrange(layer_num):

		# exit the loop after fc7 layer (skip relu7, fc8, prob)
		if i >= layer_num - 3:
			break

		name = weights[i][0,0]['name'][0]
		kind = name[:2]

		if kind == 'co':
			kernels, bias = weights[i][0,0]['weights'][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
			bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
			current = utils.conv2d_basic(current, kernels, bias)
		elif kind == 'fc':
			kernels, bias = weights[i][0,0]['weights'][0]
			kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
			bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
			current = utils.conv2d_same(current, kernels, bias)
		elif kind == 're':
			current = tf.nn.relu(current, name=name)
		elif kind == 'po':
			current = utils.max_pool_2x2(current)

		net[name] = current

	return net

def get_fc7 (image):
	"""
	Extract fc7 features from given image
	"""
	print "setting up vgg initialized conv layers ..." 
	model_dir = '/Users/deep/Programming/VGG/'
	model_name = 'vgg-face.mat'
	model_data = sio.loadmat(model_dir+model_name)
	weights = np.squeeze(model_data['layers'])

	#subracting mean
	meta = model_data['meta']
	mean =  meta[0,0]['normalization'][0,0]['averageImage']
	mean_pixel = np.mean(mean, axis=(0, 1))

	processed_image = utils.process_image(image, mean_pixel)

	image_net = vgg_net(weights, processed_image)
	fc7_layer = image_net['fc7']
	return tf.reshape(fc7_layer, [-1])
	# return fc7_layer


def merge_fc7 (features, method):
	"""
	Merge fc7 features from different images using specified method

	Inputs: 
		features: Tuple of feature vectors to be merged
		method: 'average': merge features by taking average
			'max_contrib': merge features by keeping maximal contributions

	Outputs:
		comb_feature: combined feature vector using 'method'
	"""
	comb_feature = np.vstack(features)

	if method == 'average':
		comb_feature = np.mean(comb_feature, axis=0)

	elif method == 'max_contrib':
		mask = np.argmax(np.absolute(comb_feature), axis=0)
		comb_feature = comb_feature[mask, np.arange(comb_feature.shape[1])]

	return comb_feature

def similarity (feature1, feature2, method):
	"""
	Computes similarity between two fc7 feature vectors

	Inputs: 
		feature1: feature vector1
		feature2: feature vector2
		method: 'L2': compute similarity using L2 distance
			'rank1': computer similarity using rank1 score

	Outputs:
		score: similarity score between two feature vectors
	"""

	if method == 'L2':
		score = np.sqrt(np.sum((feature1-feature2)**2, axis=1))

	elif method == 'rank1':
		pass

	return score


def main (argv=None):

	print "Start ..."
	image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_image")
	# Define placeholder for fc7 features of an image
	feature = get_fc7(image)

	# Read Images
	print "Reading images and preprocessing ..."

	# define image paths
	pairs_path = './dataset/pairsDevTrain.txt'
	suffix = 'jpg'
	root = './dataset/lfw2'

	# determine image pairs to be loaded
	pairs = load_pairs(pairs_path)

	with tf.Session() as sess:

		# init tf session and get the feature vectors for the images
		print "Evaluating forward pass for VGG face Descriptor ..." 
		sess.run(tf.global_variables_initializer())
		# define placeholders for 2 sets of images to be compared, as well as their labels
		feature1 = np.zeros([pairs.shape[0], 4096], dtype=np.float32)
		feature2 = np.zeros([pairs.shape[0], 4096], dtype=np.float32)
		same = np.zeros([pairs.shape[0]], dtype=np.int32)

		# load the images
		i=0
		for pair in pairs:
			if i%10 == 0 or i==0:
				print "Evaluated %d pairs" % (i)
			name1, name2, same[i] = pairs_info(pair, suffix)
			image1, image2 = readImage(root, name1, name2)

			feature1[i,:] = sess.run(feature, feed_dict={image: image1})
			feature2[i,:] = sess.run(feature, feed_dict={image: image2})
			i += 1

	distances = similarity(feature1, feature2, 'L2')
	file_ID = 'distances.pkl'
	f = open(file_ID, "w")
	pickle.dump(distances, f, protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(same, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

	mat = np.vstack((distances, same)).T
	sio.savemat('distances.mat', {'mat':mat})


if __name__ == "__main__":
    tf.app.run()






