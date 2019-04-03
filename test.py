import cv2
import os
import network
import config
import normalize as n
import numpy as np
import matplotlib.pyplot as plt
import confusion_matrix as cm
from PIL import Image
from chainer import serializers, Variable
from chainer.cuda import to_cpu 
from chainer import functions as F

test_img_path = config.test_img_path
test_gt_path = config.test_gt_path
save_model_path = config.save_model_path
save_predicted_path = config.save_predicted_path
img_size = config.img_size
n_class = config.n_class
max_epoch = config.max_epoch
img_normalization = config.img_normalization

palette = [
	[128,128,128],
	[128,0,0],
	[192,192,128],
	[128,64,128],
	[255,69,0],
	[60,40,222],
	[128,128,0],
	[192,128,128],
	[64,64,128],
	[64,0,128],
	[64,64,0],
	[0,128,192],
	[0,0,0]]

def draw_png(img):
	seg_img = np.zeros((img_size[0], img_size[1], 3))
	for i in range(img_size[0]):
		for j in range(img_size[1]):
			color_number = img[i,j]
			seg_img[i,j,0] = palette[color_number][0]
			seg_img[i,j,1] = palette[color_number][1]
			seg_img[i,j,2] = palette[color_number][2]
	seg_img = Image.fromarray(np.uint8(seg_img))
	
	return seg_img
	#When you use cv2, RGB will be BGR. TAKE CARE IT.

def main():
	model = network.SegNet()
	model.to_gpu(0)
	serializers.load_hdf5(os.path.join(save_model_path, "SegNet_{}.h5".format(20)), model)

	if not os.path.exists(save_predicted_path):
		os.makedirs(save_predicted_path)
		print("Made save folder")

	test_names = os.listdir(test_img_path)
	test_names.sort()
	num_imgs = len(test_names)
	imgs_eval  = np.empty((num_imgs, img_size[0], img_size[1]))

	for i in range(num_imgs):
		img = cv2.imread(os.path.join(test_img_path, test_names[i]))
		img = img.transpose(2,0,1)
		img = img.reshape((1, 3, img_size[0], img_size[1]))
		img = img.astype("float32")
		if img_normalization:
			img = n.normalize(img)
		img = Variable(img)
		img.to_gpu(0)
		seg_img = model(img)
		seg_img.to_cpu()
		seg_img = seg_img.reshape((n_class, img_size[0], img_size[1]))
		seg_img = F.argmax(seg_img, axis=0).array
		colored_seg_img = draw_png(seg_img)
		colored_seg_img.save(os.path.join(save_predicted_path, test_names[i]))
		print("{} is saved".format(test_names[i]))
		imgs_eval[i] = seg_img

	confusion_matrix, acc = cm.make_confusion_matrix(imgs_eval)
	cm.show_confusion_matrix(confusion_matrix, acc)

if __name__ == '__main__':
	main()