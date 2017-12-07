"""
Code to load the weights of pretrained VGG-Face Descriptor model and return
4096-D FC7 features.

Author: Deep Chakraborty
Date: 12/07/2017

"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio

import TensorflowUtils as utils

def vgg_net(weights, image):

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
		kind = weights[i][0,0]['type'][0]

		if kind == 'conv':
			kernels, bias = weights[i][0,0]['weights'][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
			bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
			current = utils.conv2d_basic(current, kernels, bias)
		elif kind == 'relu':
			current = tf.nn.relu(current, name=name)
		elif kind == 'pool':
			current = utils.max_pool_2x2(current)

		net[name] = current

	return net

def get_fc7(image):

	model_dir = '/Users/deep/Programming/VGG/vgg-face-tensorflow/'
	model_name = 'vgg-face.mat'
	model_data = sio.loadmat(model_dir+model_name)
	weights = np.squeeze(model_data['layers'])

	#subracting mean
	meta = model_data['meta']
	mean =  meta[0,0]['normalization'][0,0]['averageImage']
	processed_image = utils.process_image(image, mean)

	image_net = vgg_net(weights, processed_image)
	fc7_layer = image_net['fc7']

	return tf.reshape(fc7_layer, [-1])



