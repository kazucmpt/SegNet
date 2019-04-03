import cv2
import os
import network
import config
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from chainer import serializers, Variable
from chainer.cuda import to_cpu 
from chainer import functions as F

test_gt_path = config.test_gt_path
img_size = config.img_size
n_class = config.n_class

def make_confusion_matrix(pre_imgs):
	pre_imgs = pre_imgs.astype("int32")
	gt_names = os.listdir(test_gt_path)
	gt_names.sort()

	num_gts = len(gt_names)
	gt_imgs = np.empty((num_gts, img_size[0], img_size[1]), dtype="int32")
	for i in range(num_gts):
		gt_img = cv2.imread(os.path.join(test_gt_path, gt_names[i]))
		gt_img = gt_img[:, :, 0]
		gt_imgs[i] = gt_img

	confusion_matrix = np.zeros((n_class, n_class), dtype="float32")
	print("making confusion matrix...")
	for i in range(len(pre_imgs)):
		for j in range(img_size[0]):
			for k in range(img_size[1]):
				confusion_matrix[gt_imgs[i,j,k], pre_imgs[i,j,k]] += 1

	acc = np.trace(confusion_matrix)/confusion_matrix.sum()
	acc = round(acc, 3)

	for i in range(n_class):
		if confusion_matrix[i].sum() != 0:
			confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

	return confusion_matrix, acc

def show_confusion_matrix(confusion_matrix, acc):
	fig, ax = plt.subplots(figsize=(11, 8))
	heatmap = ax.pcolor(confusion_matrix, cmap=plt.cm.Blues)

	labels = ["Sky", "Building", "Pole", "Road_marking", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
	
	ax.set_xticks(np.arange(confusion_matrix.shape[0]) + 0.5, minor=False)
	ax.set_yticks(np.arange(confusion_matrix.shape[1]) + 0.5, minor=False)

	ax.invert_yaxis()
	ax.xaxis.tick_top()

	ax.set_xticklabels(labels, minor=False)
	ax.set_yticklabels(labels, minor=False)
	plt.title("Confusion Matrix acc={}".format(acc))
	plt.savefig("confusion_matrix.png")
	print("saved confusion matrix")