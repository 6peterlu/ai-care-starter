# classification code

import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten

# each sensor has ~ 1000 positive images, ~10000 negative images, and images with close indices look more similar.
SENSORS = ['02', '04', '06', '08', '10', 
		'11', '15', '21', '22', '23', 
		'24', '39', '52', '59', '62',
		'63', '72']

def load_npz(path):
	return np.load(path)['x']

def read_data_for_sensor(sensor_number, max_samples=1000000):
	data_dir = '../data/pac_data/' + sensor_number + '/'
	labels = []
	images = []
	samples_allowed = max_samples
	for label in os.listdir(data_dir)[1:]: # remove .DS_Store
		for image_filename in os.listdir(data_dir + label + '/'):
			if len(labels) < samples_allowed:
				labels.append(label)
				images.append(load_npz(data_dir + label + '/' + image_filename))
				if len(labels) % 1000 == 0:
					print('loaded ' + str(len(labels)) + ' images for sensor ' + sensor_number + ' from total ' + str(len(os.listdir(data_dir + label + '/'))))
		samples_allowed += max_samples
	labels = np.array(labels)
	images = np.expand_dims(np.array(images), 3) # adding a dimension for channels (single channel)
	print(labels.shape)
	print(images.shape)
	return labels, images

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

def create_model_resnet_50():
	model = Sequential()
	model.add(Convolution2D(3, (3, 3), activation='relu', input_shape=(240, 320, 1), data_format='channels_last'))
	model.add(ResNet50(weights='imagenet', include_top=False))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']) # TODO: what metrics?
	print('ResNet50 created')
	return model

def run_model(model, X_train, y_train, batch_size=32, epochs=10, validation_split=0.1):
	model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

labels, images = read_data_for_sensor('toy_sensor')
#model_toy_conv = create_model_toy_conv()
model_resnet_50 = create_model_resnet_50()
run_model(model_resnet_50, images, labels)

