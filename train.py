import network
import os
import cv2
import time
import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
import config
import loader
import normalize as n
from chainer import functions as F
from chainer import Variable, optimizers, serializers

max_epoch = config.max_epoch
batch_size = config.batch_size
n_class = config.n_class
save_model_path = config.save_model_path
img_normalization = config.img_normalization

class_weighting = xp.array([0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614], dtype="float32")

def draw_loss_curve(t_loss, v_loss):
	t = np.arange(0, len(t_loss))
	plt.figure(figsize=(8,8))
	plt.plot(t, t_loss, label="train", color="b")
	plt.plot(t, v_loss, label="valid", color="r")
	plt.legend(fontsize=18)
	plt.xlabel("Epoch", fontsize=18)
	plt.ylabel("Loss", fontsize=18)
	plt.title("Loss Curve", fontsize=18)
	plt.ylim(0, 2.5)
	plt.savefig("loss_curve.png")
	plt.clf()

def train():	
	if not os.path.exists(save_model_path):
		os.makedirs(save_model_path)
		print("Made save folder")

	imgs_names, gt_names, valid_imgs_names, valid_gt_names = loader.img_names_loader()

	model = network.SegNet(n_class = n_class)
	model.to_gpu(0)
	optimizer = optimizers.Adam().setup(model)

	train_loss_recode = []
	valid_loss_recode = []

	N = len(imgs_names)	
	M = len(valid_imgs_names)
	perm = np.random.permutation(N)
	perm_valid = np.random.permutation(M)
	start_time = time.time()
	for epoch in range(max_epoch):
		losses = []
		for i in range(0, N, batch_size):
			imgs_names_batch = imgs_names[perm[i:i + batch_size]]
			imgs_batch, gt_batch = loader.img_loader(imgs_names_batch)

			if img_normalization:
				imgs_batch = n.normalize(imgs_batch)

			imgs_batch = Variable(imgs_batch)
			imgs_batch.to_gpu(0)
			gt_batch = Variable(gt_batch)
			gt_batch.to_gpu(0)

			model.cleargrads()
			t = model(imgs_batch)

			loss = F.softmax_cross_entropy(t, gt_batch, class_weight=class_weighting)
			loss.backward()
			optimizer.update()

			losses.append(loss.data)

		valid_losses = []
		for i in range(0, M, batch_size):
			valid_names_batch = valid_imgs_names[perm_valid[i:i + batch_size]]
			valid_imgs_batch, valid_gt_batch = loader.img_loader(valid_names_batch, valid=True)
			if img_normalization:
				valid_imgs_batch = n.normalize(valid_imgs_batch)
			
			valid_imgs_batch = Variable(valid_imgs_batch)
			valid_imgs_batch.to_gpu(0)
			valid_gt_batch = Variable(valid_gt_batch)
			valid_gt_batch.to_gpu(0)

			model.cleargrads()
			t = model(valid_imgs_batch)
			valid_loss = F.softmax_cross_entropy(t, valid_gt_batch, class_weight=class_weighting)
			valid_losses.append(valid_loss.data)

		train_loss_recode.append(sum(losses)/len(losses))
		valid_loss_recode.append(sum(valid_losses)/len(valid_losses))
		print("epoch:{0:}\t train loss:{1:.5f}\t valid loss:{2:.5f}\t time:{3:.2f}[sec]".format(epoch, float(train_loss_recode[-1]), float(valid_loss_recode[-1]), time.time()-start_time))

		if epoch % 10 == 0:
			draw_loss_curve(train_loss_recode, valid_loss_recode)
			serializers.save_hdf5(os.path.join(save_model_path, "SegNet_{}.h5".format(epoch)), model)
			print("saved model")

if __name__ == '__main__':
	train()