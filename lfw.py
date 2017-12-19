import os
import numpy as np
from scipy.misc import imread, imresize

# The average VGG faces image
average_image = np.array([[[129.1863, 104.7624, 93.5940]]])

def load_pairs(pairs_path, root, suffix):
	print("Reading pairs...")
	pairs = []
	with open(pairs_path, 'r') as f:
		for line in f.readlines()[1:]:
			pair = line.strip().split()
			if len(pair) == 3:
				name = "{}/{}_{}.{}".format(pair[0], pair[0], '8'.zfill(4), suffix)
				if os.path.isfile(os.path.join(root, name)):
					pairs.append(pair)
			elif len(pair) == 4:
				name1 = "{}/{}_{}.{}".format(pair[0], pair[0], '4'.zfill(4), suffix)
				name2 = "{}/{}_{}.{}".format(pair[2], pair[2], '4'.zfill(4), suffix)
				if os.path.isfile(os.path.join(root, name1)) and os.path.isfile(os.path.join(root, name2)):
					pairs.append(pair)
	return np.array(pairs)

def pairs_info(pair, suffix):
	if len(pair) == 3:
		name1 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix)
		name2 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[2].zfill(4), suffix)
		same = 1
	elif len(pair) == 4:
		name1 = "{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix)
		name2 = "{}/{}_{}.{}".format(pair[2], pair[2], pair[3].zfill(4), suffix)
		same = 0
	else:
		raise Exception(
			"Unexpected pair length: {}".format(len(pair)))
	return (name1, name2, same)

def pairs_info_multiple (pair, suffix):

	name1 = []
	name2 = []
	if len(pair) == 3:
		name1.append("{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix))
		name2.append("{}/{}_{}.{}".format(pair[0], pair[0], pair[2].zfill(4), suffix))
		for i in range(8):
			num = str(i+1)
			if num != pair[1] and num != pair[2]:
				if len(name1) < 4:
					name1.append("{}/{}_{}.{}".format(pair[0], pair[0], num.zfill(4), suffix))
				elif len(name2) < 4:
					name2.append("{}/{}_{}.{}".format(pair[0], pair[0], num.zfill(4), suffix))
		same = 1

	elif len(pair) == 4:
		name1.append("{}/{}_{}.{}".format(pair[0], pair[0], pair[1].zfill(4), suffix))
		name2.append("{}/{}_{}.{}".format(pair[2], pair[2], pair[3].zfill(4), suffix))
		for i in range(4):
			num = str(i+1)
			if len(name1) < 4 and num != pair[1]:
				name1.append("{}/{}_{}.{}".format(pair[0], pair[0], num.zfill(4), suffix))
			if len(name2) < 4 and num != pair[3]:
				name2.append("{}/{}_{}.{}".format(pair[2], pair[2], num.zfill(4), suffix))
		same = 0

	else:
		raise Exception(
			"Unexpected pair length: {}".format(len(pair)))
	return (name1, name2, same)


def readImage(root, name1, name2):
	im1 = imread(os.path.join(root, name1))
	im2 = imread(os.path.join(root, name2))

	# convert RGB images to BGR
	# im1 = im1[:,:,[2,1,0]]
	# im2 = im2[:,:,[2,1,0]]

	# resize images down to 224x224 for VGG
	im1 = imresize(im1, (224, 224))
	im2 = imresize(im2, (224, 224))

	# subtract average VGG faces image
	im1 = im1 - average_image
	im2 = im2 - average_image

	# reshape images to batch size of 1
	im1 = im1.reshape(1, 224, 224, 3)
	im2 = im2.reshape(1, 224, 224, 3)
	
	return im1, im2

