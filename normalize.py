import cv2
import os
import config
import numpy as np

img_size = config.img_size

def normalize(imgs):
	num_imgs = len(imgs)
	imgs = imgs.astype("uint8")
	normalized_imgs = np.zeros((num_imgs, 3, img_size[0], img_size[1]), dtype="float32")
	for i in range(num_imgs):
		normalized_imgs[i,0,:,:] = cv2.equalizeHist(imgs[i,0,:,:])
		normalized_imgs[i,1,:,:] = cv2.equalizeHist(imgs[i,1,:,:])
		normalized_imgs[i,2,:,:] = cv2.equalizeHist(imgs[i,2,:,:])

	return normalized_imgs