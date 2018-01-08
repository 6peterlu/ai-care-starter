# scripts to visualize images and history plots

import numpy as np
import cv2
import os
import pickle
import matplotlib
matplotlib.use('TkAgg') # literally what (https://stackoverflow.com/questions/4130355/python-matplotlib-framework-under-macosx)
# all matplotlib imports need to be below this line, but I don't know why
import matplotlib.pyplot as plt

IMAGE_SRC = '../data/pac_data/02/0/20170109_142713_808.npz'
IMAGE_DEPTH_MAP = np.load(IMAGE_SRC)['x']

MODEL_HISTORY = '../saves/model_history'

print('image shape: ' + str(IMAGE_DEPTH_MAP.shape))
print('image type: ' + str(IMAGE_DEPTH_MAP.dtype))

# function from paper, normalizes real values into a 255 RGB space
def depth_map_to_image(depth_map):
	float_32_depth_map = depth_map.astype(np.float32) # converting to float32 required for cv2.normalize
	img = cv2.normalize(float_32_depth_map, float_32_depth_map, 0, 1, cv2.NORM_MINMAX)
	img = np.array(img * 255, dtype=np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
	return img

def show_ocean(data):
	print('showing ocean image')
	cv2.imshow("Ocean Image", depth_map_to_image(data))
	cv2.waitKey(0) # displays image until any key is pressed


def plot_loss(history_file):
	history = pickle.load(open(history_file, "rb"))
	# accuracy plot
	plt.plot(history['binary_accuracy'])
	plt.plot(history['val_binary_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	#loss plot
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

# method will fail if num_samples > number of examples in a folder
def generate_ocean_sample(sensor_id, num_samples):
	dir_root = '../data/pac_data/' + sensor_id + '/'
	os.makedirs('../data/sensor' + sensor_id + 'oceans/')
	selected_neg = np.random.choice(os.listdir(dir_root + '0/'), num_samples)
	selected_pos = np.random.choice(os.listdir(dir_root + '1/'), num_samples)
	for example in selected_neg:
		depth_map = np.load(dir_root + '0/' + example)['x']
		img = depth_map_to_image(depth_map)
		cv2.imwrite('../data/sensor' + sensor_id + 'oceans/' + example[:-4] + '.jpg', img)
	for example in selected_pos:
		depth_map = np.load(dir_root + '1/' + example)['x']
		img = depth_map_to_image(depth_map)
		cv2.imwrite('../data/sensor' + sensor_id + 'oceans/' + example[:-4] + '.jpg', img)

#plot_loss(MODEL_HISTORY)
#show_ocean(IMAGE_DEPTH_MAP)
#print(IMAGE_DEPTH_MAP)

plot_loss(MODEL_HISTORY)