import os
import cv2
import numpy as np
import config

img_size = config.img_size
train_img_path = config.train_img_path
valid_img_path = config.valid_img_path
train_gt_path = config.train_gt_path
valid_gt_path = config.valid_gt_path

def img_loader(imgs_names, valid=False):
	num_imgs = len(imgs_names)

	imgs = np.empty((num_imgs, 3, img_size[0], img_size[1]), dtype="float32")
	for i in range(num_imgs):
		if not valid:
			img = cv2.imread(os.path.join(train_img_path, imgs_names[i]))
		else:
			img = cv2.imread(os.path.join(valid_img_path, imgs_names[i]))
		img = img.transpose(2,0,1)
		img = img.astype("float32")
		imgs[i] = img

	gts = np.empty((num_imgs, img_size[0], img_size[1]), dtype="int32")
	for i in range(num_imgs):
		if not valid:
			gt = cv2.imread(os.path.join(train_gt_path, imgs_names[i]))
		else:
			gt = cv2.imread(os.path.join(valid_gt_path, imgs_names[i]))
		gt = gt.astype("int32")
		gt = gt[:,:,0] #gt[:,:,0] == gt[:,:,1] == gt[:,:,2]
		gt = gt.reshape(img_size[0], img_size[1]) #which is better it or flatten()?
		gts[i] = gt

	"""
	gts = np.empty((num_imgs, img_size[0]*img_size[1]), dtype="int32")
	for i in range(num_imgs):
		if not valid:
			gt = cv2.imread(os.path.join(train_gt_path, imgs_names[i]))
		else:
			gt = cv2.imread(os.path.join(valid_gt_path, imgs_names[i]))
		gt = gt.astype("int32")
		gt = gt[:,:,0] #gt[:,:,0] == gt[:,:,1] == gt[:,:,2]
		gt = gt.reshape(img_size[0]*img_size[1]) #which is better it or flatten()?
		gts[i] = gt
	"""

	return imgs, gts

def img_names_loader():
	imgs_names = os.listdir(train_img_path)
	imgs_names.sort()
	imgs_names = np.asarray(imgs_names)

	gt_names = os.listdir(train_gt_path)
	gt_names.sort()
	gt_names = np.asarray(gt_names)

	valid_imgs_names = os.listdir(valid_img_path)
	valid_imgs_names.sort()
	valid_imgs_names = np.asarray(valid_imgs_names)

	valid_gt_names = os.listdir(valid_gt_path)
	valid_gt_names.sort()
	valid_gt_names = np.asarray(valid_gt_names)

	return imgs_names, gt_names, valid_imgs_names, valid_gt_names