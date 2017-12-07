from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils

def vgg_net(weights, image):
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

		'fc7'
	)

for i, name in enumerate(layers):
	kind = name[:2]
	if kind == 'co' or kind == 'fc':
		kernels, bias = weights[i][0][0][2][0]
		# matconvnet: weights are [width, height, in_channels, out_channels]
		# tensorflow: weights are [height, width, in_channels, out_channels]
		print i, name, kernels.shape, bias.shape
		#         kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
		#         bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
		#         current = utils.conv2d_basic(current, kernels, bias)
	elif kind == 're':
		print i, name
	#         current = tf.nn.relu(current, name=name)
	elif kind == 'po':
		print i, name
	#         current = utils.avg_pool_2x2(current)
	#     net[name] = current

	return net