# classification code

import numpy as np
import os
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model

# each sensor has ~ 1000 positive images, ~10000 negative images, and images with close indices look more similar.
SENSORS = ['02', '04', '06', '08', '10', 
		'11', '15', '21', '22', '23', 
		'24', '39', '52', '59', '62',
		'63', '72']

def load_npz(path):
	return np.load(path)['x']

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

def read_data_for_sensor(sensor_number, max_samples=1000000):
	np.random.seed(231)
	data_dir = '../data/pac_data/' + sensor_number + '/'
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

def sample_from_all_sensors(max_samples=10000000):
	data_dir = '../data/pac_data/'
	all_files, all_labels = load_all_file_paths(data_dir)
	images = []
	for file in all_files:
		if len(images) >= max_samples:
			continue
		images.append(load_npz(file))
	images = np.expand_dims(np.array(images), axis=3)
	labels = all_labels[:max_samples]
	labels = np.array(labels)
	print(images.shape)
	print(labels.shape)
	return images, labels


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

def train_model(model, X_train, y_train, batch_size=100, epochs=30, validation_split=0.2):
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
	model.save('../saves/model_data.h5') # save weights
	with open('../saves/model_history', 'wb') as history_file: # save loss history
		pickle.dump(history.history, history_file)

def evaluate_model(model, X_test, y_test, batch_size=50):
	score = model.evaluate(X_test, y_test, batch_size=batch_size)
	print(score)
	
images, labels = sample_from_all_sensors(max_samples = 50)
model_resnet_50 = create_model_resnet_50()
train_model(model_resnet_50, images, labels)

#labels, images = read_data_for_sensor('02', max_samples=1000)
#model_toy_conv = create_model_toy_conv()
#model_resnet_50 = create_model_resnet_50()
#train_model(model_resnet_50, images, labels)
#model = load_model('../saves/model_data.h5')
#evaluate_model(model, images, labels)




