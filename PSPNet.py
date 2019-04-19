import chainer
import chainer.links as L 
import chainer.functions as F
from chainer import Chain
import math

import config

n_class = config.n_class
LRN = config.LRN
dropout = config.dropout

class PSPNet(Chain):

	def __init__(self, n_class=n_class):
		super().__init__()
		with self.init_scope():

			self.n_class = n_class

			#Encoder
			self.enco1_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.enco1_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.enco2_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.enco2_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)	
			self.enco3_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.enco3_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1) 
			
			#Pyramid Module
			self.pconv1 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True)
			self.pconv2 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True)
			self.pconv3 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True)
			self.pconv4 = L.Convolution2D(None, 256, ksize=1, stride=1, pad=0, nobias=True)

			#Decoder
			self.deco6_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.deco6_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.deco7_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.deco7_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.deco8_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.deco8_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)

			#Fianal Layer
			self.final_layer = L.Convolution2D(None, n_class, ksize=1, stride=1, pad=0, nobias=True)

			#BatchNormalizations
			self.bn1_1 = L.BatchNormalization(  64)
			self.bn1_2 = L.BatchNormalization(  64)
			self.bn2_1 = L.BatchNormalization( 128)
			self.bn2_2 = L.BatchNormalization( 128)
			self.bn3_1 = L.BatchNormalization( 256)
			self.bn3_2 = L.BatchNormalization( 256)

			self.bnp1 = L.BatchNormalization(256)
			self.bnp2 = L.BatchNormalization(256)
			self.bnp3 = L.BatchNormalization(256)
			self.bnp4 = L.BatchNormalization(256)
			self.bnp5 = L.BatchNormalization(256)

			self.bn6_1 = L.BatchNormalization( 256)
			self.bn6_2 = L.BatchNormalization( 256)
			self.bn7_1 = L.BatchNormalization( 128)
			self.bn7_2 = L.BatchNormalization( 128)
			self.bn8_1 = L.BatchNormalization(  64)
			self.bn8_2 = L.BatchNormalization(  64)

	def __call__(self, x): #x = (batchsize, 3, 360, 480)
		
		#Encode Module
		h1_1 = F.relu(self.bn1_1(self.enco1_1(x)))
		h1_2 = F.relu(self.bn1_2(self.enco1_2(h1_1)))
		pool1 = F.max_pooling_2d(h1_2, ksize=2, stride=2, return_indices=False) #(batchsize,  64, 180, 240)
		h2_1 = F.relu(self.bn2_1(self.enco2_1(pool1)))
		h2_2 = F.relu(self.bn2_2(self.enco2_2(h2_1)))
		pool2 = F.max_pooling_2d(h2_2, ksize=2, stride=2, return_indices=False) #(batchsize, 128,  90, 120) 
		h3_1 = F.relu(self.bn3_1(self.enco3_1(pool2)))
		h3_2 = F.relu(self.bn3_2(self.enco3_2(h3_1)))
		encoded_x = F.max_pooling_2d(h3_2, ksize=2, stride=2, return_indices=False) #(batchsize, 256,  45,  60) 

		_, _, h, w = encoded_x.data.shape

		#Pyramid Pooling Module
		x1 = F.average_pooling_2d(encoded_x, ksize=1, stride=1)
		x1 = self.pconv1(x1)
		x1 = self.bnp1(x1)
		x1 = F.resize_images(x1, (h, w))

		x2 = F.average_pooling_2d(encoded_x, ksize=2, stride=2)
		x2 = self.pconv2(x2)
		x2 = self.bnp2(x2)
		x2 = F.resize_images(x2, (h, w))

		x3 = F.average_pooling_2d(encoded_x, ksize=3, stride=3)
		x3 = self.pconv3(x3)
		x3 = self.bnp3(x3)
		x3 = F.resize_images(x3, (h, w))

		x4 = F.average_pooling_2d(encoded_x, ksize=6, stride=6)
		x4 = self.pconv4(x4)
		x4 = self.bnp4(x4)
		x4 = F.resize_images(x4, (h, w))

		concated_x = F.concat((encoded_x, x1, x2, x3, x4))
		
		#Decode Module
		up5 = F.unpooling_2d(concated_x, ksize=2, stride=2, outsize=(pool2.shape[2], pool2.shape[3]))
		h6_1 = F.relu(self.bn6_1(self.deco6_1(F.concat((up5, h3_2)))))
		h6_2 = F.relu(self.bn6_2(self.deco6_2(h6_1)))

		up6 = F.unpooling_2d(h6_2, ksize=2, stride=2, outsize=(pool1.shape[2], pool1.shape[3]))
		h7_1 = F.relu(self.bn7_1(self.deco7_1(F.concat((up6, h2_2)))))
		h7_2 = F.relu(self.bn7_2(self.deco7_2(h7_1)))

		up7 = F.unpooling_2d(h7_2, ksize=2, stride=2, outsize=(x.shape[2], x.shape[3]))
		h8_1 = F.relu(self.bn8_1(self.deco8_1(F.concat((up7, h1_2)))))
		h8_2 = F.relu(self.bn8_2(self.deco8_2(h8_1)))

		h = self.final_layer(h8_2)
		return h
