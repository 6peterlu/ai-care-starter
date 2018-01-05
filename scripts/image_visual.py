# scripts to visualize images

import numpy as np
import cv2

IMAGE_SRC = '../data/pac_data/02/0/20160930_082811_796.npz'
IMAGE_DEPTH_MAP = np.load(IMAGE_SRC)['x']

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

show_ocean(IMAGE_DEPTH_MAP)