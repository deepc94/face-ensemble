"""
Code to load the weights of pretrained VGG-Face Descriptor model and return
4096-D FC7 features.

Author: Deep Chakraborty
Date: 12/07/2017
"""

__author__ = "Deep Chakraborty"

import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.misc import imread, imsave, imresize

import TensorflowUtils as utils

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
	processed_image = utils.process_image(image, mean)

	image_net = vgg_net(weights, processed_image)
	fc7_layer = image_net['fc7']
	#TODO: Debug shape of fc7_layer (200704,)
	# return tf.reshape(fc7_layer, [-1])
	return fc7_layer


def merge_fc7 (features, method):
	"""
	TODO: Merge fc7 features from different images using specified method

	Inputs: 
		features: List of feature vectors to be merged
		method: 'average': merge features by taking average
			'argmax': merge features by keeping argmax

	Outputs:
		comb_feature: combined feature vector using 'method'
	"""
	pass

def similarity (feature1, feature2, method, threshold):
	"""
	TODO: Computes similarity between two fc7 feature vectors

	Inputs: 
		feature1: feature vector1
		feature2: feature vector2
		method: 'L2': compute similarity using L2 distance
			'rank1': computer similarity using rank1 score
		threshold: similarity threshold for reporting same or not

	Outputs:
		score: similarity score between two feature vectors
		same: whether features belong to same person or not 
			(True or False)
	"""
	pass

def main (argv=None):

	image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_image")
	feature = get_fc7(image)

	# Read Images
	print "Reading images and preprocessing ..."
	im1 = imread('/Users/deep/Programming/VGG/lfw2/Aaron_Sorkin/Aaron_Sorkin_0001.jpg')
	im2 = imread('/Users/deep/Programming/VGG/lfw2/Aaron_Sorkin/Aaron_Sorkin_0002.jpg')
	im3 = imread('/Users/deep/Programming/VGG/lfw2/Frank_Solich/Frank_Solich_0005.jpg')
	im4 = imread('/Users/deep/Programming/VGG/lfw2/Frank_Solich/Frank_Solich_0004.jpg')

	# convert RGB images to BGR
	im1 = im1[:,:,[2,1,0]]
	im2 = im2[:,:,[2,1,0]]
	im3 = im3[:,:,[2,1,0]]
	im4 = im4[:,:,[2,1,0]]

	# resize images down to 224x224
	im1 = imresize(im1, (224, 224)).reshape((1, 224, 224, 3))
	im2 = imresize(im2, (224, 224)).reshape((1, 224, 224, 3))
	im3 = imresize(im3, (224, 224)).reshape((1, 224, 224, 3))
	im4 = imresize(im4, (224, 224)).reshape((1, 224, 224, 3))

	# init tf session and get the feature vectors for the 4 images
	print "Evaluating forward pass for VGG face Descriptor ..." 
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	feat1 = sess.run(feature, feed_dict={image: im1})
	feat2 = sess.run(feature, feed_dict={image: im2})
	feat3 = sess.run(feature, feed_dict={image: im3})
	feat4 = sess.run(feature, feed_dict={image: im4})

	print feat1.shape 
	print feat2.shape
	print feat3.shape
	print feat4.shape 

if __name__ == "__main__":
    tf.app.run()






