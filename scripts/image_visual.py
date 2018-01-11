# scripts to visualize images and history plots

import numpy as np
import cv2
import os
import pickle
#import matplotlib
#matplotlib.use('TkAgg') # literally what (https://stackoverflow.com/questions/4130355/python-matplotlib-framework-under-macosx)
# all matplotlib imports need to be below this line, but I don't know why
#import matplotlib.pyplot as plt

IMAGE_SRC = '../data/augmented/02/1/20170109_182454_820.npz'

#IMAGE_DEPTH_MAP = np.load(IMAGE_SRC)

MODEL_HISTORY = '../saves/model_history'

SENSORS = ['02', '04', '06', '08', '10', 
		'11', '15', '21', '22', '23', 
		'24', '39', '52', '59', '62',
		'63', '72']
BOUNDING_BOXES = [(56,240,144,239),
		(155,292,67,144),
		(195,316,173,239),
		(152,279,57,140),
		(0,319,0,239),
		(46,202,165,239),
		(158,276,94,181),
		(132,268,79,155),
		(0,319,0,239),
		(0,319,0,239),
		(0,319,0,239),
		(139,297,151,240)
		(133,307,132,233),
		(80,241,145,233),
		(66,189,145,228),
		(10,142,112,207),
		(168,318,57,153)]

# print('image shape: ' + str(IMAGE_DEPTH_MAP.shape))
# print('image type: ' + str(IMAGE_DEPTH_MAP.dtype))

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
	print(history['val_binary_accuracy'])
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
	dir_root = '../data/augmented/' + sensor_id + '/'
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

# bounding box (x_min, x_max, y_min, y_max)
def generate_bounding_box_segment(sensor_id, bounding_box):
	x_min, x_max, y_min, y_max = bounding_box
	dir_root = '../data/pac_data/' + sensor_id + '/'
	neg_dir = '../data/augmented/' + sensor_id + '/0/'
	pos_dir = '../data/augmented/' + sensor_id + '/1/'
	os.makedirs(neg_dir)
	os.makedirs(pos_dir)
	for file in os.listdir(dir_root + '0/'):
		full_file = dir_root + '0/' + file
		depth_map = np.load(full_file)['x']
		original = np.copy(depth_map)
		outfile = open(neg_dir + file, 'w')
		depth_map[:y_min, :] = 0.0
		depth_map[y_max, :] = 0.0
		depth_map[:, :x_min] = 0.0
		depth_map[:, x_max:] = 0.0
		concat = np.concatenate((depth_map, original), axis=0)
		np.savez(neg_dir + file[:-4], x=concat)
		outfile.close()
	if (os.path.isdir(dir_root + '1/')):
		for file in os.listdir(dir_root + '1/'):
			full_file = dir_root + '1/' + file
			depth_map = np.load(full_file)['x']
			original = np.copy(depth_map)
			outfile = open(pos_dir + file, 'w')
			depth_map[:y_min, :] = 0.0
			depth_map[y_max, :] = 0.0
			depth_map[:, :x_min] = 0.0
			depth_map[:, x_max:] = 0.0
			concat = np.concatenate((depth_map, original), axis=0)
			np.savez(pos_dir + file[:-4], x=concat)
			outfile.close()

def generate_all_bounding_boxes(sensors, bounding_boxes):
	for i in range(len(sensors)):
		generate_bounding_box_segment(sensors[i], bounding_boxes[i])
#plot_loss(MODEL_HISTORY)
#show_ocean(IMAGE_DEPTH_MAP)
#print(IMAGE_DEPTH_MAP)

plot_loss(MODEL_HISTORY)

generate_all_bounding_boxes(SENSORS, BOUNDING_BOXES)
