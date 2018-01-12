# classification code

import numpy as np
import os
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import svm


# each sensor has ~ 1000 positive images, ~10000 negative images, and images with close indices look more similar.
SENSORS = ['02', '04', '06', '08', '10', 
		'11', '15', '21', '22', '23', 
		'24', '39', '52', '59', '62',
		'63', '72']

DATA_VERSION = 'augmented' # augmented or pac_data

# helper method to load an individual npz file
def load_npz(path):
	return np.load(path)['x']

# NOTE: this method shuffles the files already
# Assembles a list of all files in the root directory (either augmented or pac_data)
def load_all_file_paths(directory):
	np.random.seed(231)
	all_files = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			if f != '.DS_Store':
				all_files.append(os.path.join(dirpath, f))
		print('dirpath: ' + dirpath + ' complete')
	np.random.shuffle(all_files)
	all_labels = []
	for file in all_files:
		if (DATA_VERSION == 'augmented'):
			all_labels.append(int(file[21:22])) # hardcoded based on current directory structure
		else:
			all_labels.append(int(file[20:21])) # hardcoded based on current directory structure
	return all_files, all_labels

# reads in max_samples of data for a single sensor. Not used in the final resnet.
def read_data_for_sensor(sensor_number, max_samples=1000000):
	np.random.seed(231)
	data_dir = '../data/' + DATA_VERSION + '/' + sensor_number + '/'
	labels = []
	images = []
	samples_allowed = max_samples
	for label in os.listdir(data_dir): # remove .DS_Store
		if label == '.DS_Store':
			continue
		files_list = os.listdir(data_dir + label + '/')
		np.random.shuffle(files_list) # shuffle so similar samples are not in the same set
		for image_filename in files_list:
			if len(labels) < samples_allowed:
				labels.append(label)
				images.append(load_npz(data_dir + label + '/' + image_filename))
				if len(labels) % 1000 == 0:
					print('loaded ' + str(len(labels)) + ' images for sensor ' + sensor_number + ' from total ' + str(len(os.listdir(data_dir + label + '/'))))
		samples_allowed += max_samples
	labels = np.array(labels)
	images = np.expand_dims(np.array(images), axis=3) # adding a dimension for channels (single channel)
	print(labels.shape) # ? * 1
	print(images.shape) # ? * 240 * 320 * 1
	return images, labels

# SVMs require 2D data, so this method reshapes the matrices for unconcatenated images.
def convert_data_to_svm_format(X_train, X_test):
	X_train_svm = np.reshape(X_train, (-1, 240 * 320))
	X_test_svm = np.reshape(X_test, (-1, 240 * 320))
	print(X_train_svm.shape)
	print(X_test_svm.shape)
	return X_train_svm, X_test_svm

# sampling data from all sensors with a set pos-neg and train-test split.
def sample_from_all_sensors(max_samples=1000, train_split=0.8, pos_split=0.4):
	np.random.seed(231)
	data_dir = '../data/' + DATA_VERSION + '/'
	all_files, all_labels = load_all_file_paths(data_dir)
	positive_files = []
	negative_files = []
	for i in range(len(all_labels)):
		if all_labels[i] == 1:
			positive_files.append(all_files[i])
		else:
			negative_files.append(all_files[i])
	print(len(negative_files))
	print(len(positive_files))
	pos_split_index = int(pos_split * max_samples)
	neg_split_index = int((1 - pos_split) * max_samples)
	sample_files = positive_files[:pos_split_index] + negative_files[:neg_split_index]

	images = []
	for file in sample_files:
		if len(images) >= max_samples:
			continue
		images.append(load_npz(file))
	images = np.expand_dims(np.array(images), axis=3)
	labels = [1.0] * pos_split_index + [0.0] * neg_split_index
	labels = np.array(labels)

	# shuffling positives and negatives
	shuffle_indices = np.random.permutation(len(labels))
	labels = labels[shuffle_indices]
	images = images[shuffle_indices, :, :, :]

	# creating train test splits
	split_index = int(train_split * max_samples)
	train_images = images[:split_index, :, :, :]
	train_labels = labels[:split_index]
	test_images = images[split_index:, :, :, :]
	test_labels = labels[split_index:]
	print(np.sum(train_labels))
	print(len(train_labels))
	print(np.sum(test_labels))
	print(len(test_labels))
	return train_images, train_labels, test_images, test_labels


# a tiny convolutional model to test data pipeline
def create_model_toy_conv():
	model = Sequential()
	model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(240, 320, 1), data_format='channels_last'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']) # TODO: what metrics?
	print('Toy conv net created')
	return model

# a resnet taking in regular image input
def create_model_resnet_50():
	model = Sequential()
	model.add(Convolution2D(3, (3, 3), activation='relu', input_shape=(240, 320, 1), data_format='channels_last'))
	model.add(ResNet50(weights=None, include_top=False))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']) # TODO: what metrics?
	print('ResNet50 created')
	return model

# a resnet taking in concatenated image input
def create_model_concat_resnet_50():
	model = Sequential()
	model.add(Convolution2D(3, (3, 3), activation='relu', input_shape=(480, 320, 1), data_format='channels_last'))
	model.add(ResNet50(weights=None, include_top=False))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']) # TODO: what metrics?
	print('concat ResNet50 created')
	return model

# a toy affine model to check data pipeline success
def create_model_toy_affine():
	model = Sequential()
	model.add(Flatten(input_shape=(240, 320, 1)))
	model.add(Dense(1024))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	print('toy affine model created')
	return model

# trains and evaluates the vanilla SVM
def train_svm(X_train, y_train, X_test, y_test, is_testing=True):
	clf = svm.SVC(gamma=0.001, C=100)
	clf.fit(X_train, y_train)
	if (is_testing):
		predicted = clf.predict(X_train)
		print(predicted)
		print(y_train)
		print(np.sum(predicted * y_train)/np.sum(y_train))
		predicted_test = clf.predict(X_test)
		print(np.sum(predicted_test * y_test)/np.sum(y_test))

# trains the model, and saves the configuration at the end of training
def train_model(model, X_train, y_train, batch_size=100, epochs=40, validation_split=0.5):
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
	model.save('../saves/concat_resnet_dropout_data.h5') # save weights
	with open('../saves/concat_resnet_dropout_history', 'wb') as history_file: # save loss history
		pickle.dump(history.history, history_file)

# generates ROC curve data, can be plotted in image_visual.py
def run_model(model, X_test, y_test):
	prediction = model.predict(X_test)
	roc_auc = roc_auc_score(y_test, prediction)
	print(roc_auc)
	fpr, tpr, _ = roc_curve(y_test, prediction)
	np.save('../roc/fpr', fpr)
	np.save('../roc/tpr', tpr)

# generates a single accuracy metric
def evaluate_model(model, X_test, y_test, batch_size=100):
	score = model.evaluate(X_test, y_test, batch_size=batch_size)
	print(score)

# STEP 1: GENERATE AUGMENTED DATA IN image_visual.py

# STEP 2: TRAIN MODEL ON AUGMENTED DATA
train_images, train_labels, test_images, test_labels = sample_from_all_sensors(max_samples = 2000)
model_concat_resnet_50 = create_model_concat_resnet_50()
train_model(model_concat_resnet_50, train_images, train_labels)
run_model(model_concat_resnet_50, test_images, test_labels)

# STEP 3: REVIEW RESULTS OF DATA IN image_visual.py




