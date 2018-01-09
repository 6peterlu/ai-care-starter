from keras import applications
from keras import backend as K
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers, optimizers, metrics
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from PIL import Image
from PIL import ImageFile
import os
import sys
import subprocess
import csv
import pandas as pd
import h5py
import tensorflow as tf
import glob
from os import path

LOG_PATH = '../data/logs'

num_epochs = 5
train_batch_size = 16
test_batch_size = 10
reg = 1e-5
lr = 1e-4

def plot():
	pass

SENSORS = ['02', '04', '06', '08', '10', 
		'11', '15', '21', '22', '23', 
		'24', '39', '52', '59', '62',
		'63', '72']

def load_npz(path):
	temp = np.stack((np.load(path)['x'], np.load(path)['x'], np.load(path)['x']), axis=-1)
	print(temp.shape)
	return temp

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
		all_labels.append(int(file[20:21])) # hardcoded based on current directory structure
	return all_files, all_labels

def sample_from_all_sensors(max_samples=1000, train_split=0.8, pos_split=0.4):
	np.random.seed(231)
	data_dir = '../data/pac_data/'
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
	labels = [1.0] * pos_split_index + [0.0] * neg_split_index
	labels = np.array(labels)

	# shuffling positives and negatives
	shuffle_indices = np.random.permutation(len(labels))
	labels = labels[shuffle_indices]
	print("shape")
	images = np.array(images)
	print(images.shape)
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


def mixed_ce_objective(y_true, y_pred):
	print("y true", y_true)
	binary_ce = K.mean(K.sum(K.binary_crossentropy(y_true, y_pred), 1))
	print("binary", binary_ce)
	return binary_ce

def create_model():
	print('Loaded Pre-Trained Model')
	resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(240, 320, 3))
	x = Flatten()(resnet50.output)

	preds = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(reg))(x)

	model = Model(inputs=resnet50.input, outputs=preds)
	adam = optimizers.Adam(lr=lr)
	model.compile(loss=mixed_ce_objective, optimizer=adam)
	return model


def main():
	train_images, train_labels, test_images, test_labels = sample_from_all_sensors(max_samples = 200)
	
	X_train = train_images
	print(X_train.shape)
	y_train = train_labels
	model = create_model()
    # model.load_weights('model_checkpoint')
	tensorboard = TensorBoard(log_dir=LOG_PATH, write_graph=True, write_images=True)
	checkpoint = ModelCheckpoint(filepath='./model_checkpoint', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	model.fit(X_train, y_train, validation_split=0.2, epochs=num_epochs, batch_size=train_batch_size, callbacks=[tensorboard, checkpoint])
	
	X_test = test_images
	y_test = test_labels
	prediction = model.predict(X_test)
	print(prediction)
	f = open('predictions_test.txt','w')
	for image in prediction:
		for i in range(len(image)):
			if i != (len(image)-1):
				f.write("%s, " % image[i])
			else:
				f.write("%s" % image[i])
		f.write("\n")


if __name__ == '__main__':
	main()

