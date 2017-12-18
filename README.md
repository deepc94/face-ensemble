# face-ensemble
Try to achieve better separation between matched and mismatched faces from LFW dataset through maximal contribution from multiple face images per person instead of only one image per person.

### Description of files
* `main.py`: Main file to load the VGG pre-trained model and training images to extract fc7 features, compute distance between face features and save them.
* `lfw.py`: utility to load the lfw development training set
* `TensorflowUtils.py`: Utility for different types of layers in the CNN
* `Dataset`: Folder containing the lfw2 dataset and metadata txts.
* `Results`: Folder containing any histogram images, other results, etc.

#### Description of functions in `main.py`:
* `vgg_net (weights, image)`: Performs a forward pass on the input image and returns the fc7 layer
* `get_fc7 (image)`: Loads the model weights, preprocesses the input image and calls `vgg_net` to extract the fc7 features for that image
* `merge_fc7 (features, method)`: Merges the given fc7 feature vectors into one by using `average` or `max_contrib` methods
* `similarity (feature1, feature2, method)`: Computes the similarity between two feature vectors using `L2` or `Rank1` distance
* `main ()`: Loads the dataset, creates tensorflow session and gets the feature vectors for training images, optionally combines feature vectors, computes the similarity between feature pairs and saves the distance matrix for plotting histograms.

### How to Run
* Edit the `model_dir` and `model_name` in `main.py` to reflect the path and name of the pretrained VGG model
* Download the lfw2 dataset and extract it to the dataset folder
* Dependencies: Numpy, Scipy, Tensorflow
* Run `python main.py` to generate a distance matrix and save it in pickle and mat format
