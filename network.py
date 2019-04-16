import chainer
import chainer.links as L 
import chainer.functions as F
import config
from chainer import Chain

n_class = config.n_class
LRN = config.LRN
dropout = config.dropout

class SegNet(Chain):

	def __init__(self, n_class=n_class):
		super().__init__()
		with self.init_scope():

			self.n_class = n_class

			self.enco1_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1) #image size will not change
			self.enco2_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.enco3_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.enco4_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)

			self.deco4_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
			self.deco3_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.deco2_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.deco1_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.deco0_1 = L.Convolution2D(None, n_class, ksize=1)

			self.bn_enco1_1 = L.BatchNormalization( 64)
			self.bn_enco2_1 = L.BatchNormalization(128)
			self.bn_enco3_1 = L.BatchNormalization(256)
			self.bn_enco4_1 = L.BatchNormalization(512)

			self.bn_deco4_1 = L.BatchNormalization(512)
			self.bn_deco3_1 = L.BatchNormalization(256)
			self.bn_deco2_1 = L.BatchNormalization(128)
			self.bn_deco1_1 = L.BatchNormalization( 64)

	def __call__(self, x): #x = (batchsize, 3, 360, 480)
		if LRN:
			x = F.local_response_normalization(x) #Needed for preventing from overfitting

		h = F.relu(self.bn_enco1_1(self.enco1_1(x)))
		h, idx1 = F.max_pooling_2d(h, 2, stride=2, return_indices=True)
		h = F.relu(self.bn_enco2_1(self.enco2_1(h)))
		h, idx2 = F.max_pooling_2d(h, 2, stride=2, return_indices=True) 
		h = F.relu(self.bn_enco3_1(self.enco3_1(h)))
		h, idx3 = F.max_pooling_2d(h, 2, stride=2, return_indices=True) 
		h = F.relu(self.bn_enco4_1(self.enco4_1(h)))
		if dropout:
			h = F.dropout(h)
		h, idx4 = F.max_pooling_2d(h, 2, stride=2, return_indices=True) 

		h = F.relu(self.bn_deco4_1(self.deco4_1(h)))
		h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(idx3.shape[2], idx3.shape[3]))
		h = F.relu(self.bn_deco3_1(self.deco3_1(h)))
		h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(idx2.shape[2], idx2.shape[3]))
		h = F.relu(self.bn_deco2_1(self.deco2_1(h)))
		h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(idx1.shape[2], idx1.shape[3]))
		h = F.relu(self.bn_deco1_1(self.deco1_1(h)))
		h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(x.shape[2], x.shape[3]))

		h = self.deco0_1(h)

		return h

class UNet(Chain):

	def __init__(self, n_class=n_class):
		super().__init__()
		with self.init_scope():

			self.n_class = n_class

			self.enco1_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.enco1_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.enco2_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.enco2_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)	
			self.enco3_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.enco3_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)	
			self.enco4_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
			self.enco4_2 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
			self.enco5_1 = L.Convolution2D(None,1012, ksize=3, stride=1, pad=1)
			self.deco6_1 = L.Convolution2D(None,1012, ksize=3, stride=1, pad=1)
			self.deco6_2 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
			self.deco7_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
			self.deco7_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.deco8_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
			self.deco8_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.deco9_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
			self.deco9_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
			self.deco9_3 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)

			self.final_layer = L.Convolution2D(None, n_class, ksize=1)

			self.bn1_1 = L.BatchNormalization(  64)
			self.bn1_2 = L.BatchNormalization(  64)
			self.bn2_1 = L.BatchNormalization( 128)
			self.bn2_2 = L.BatchNormalization( 128)
			self.bn3_1 = L.BatchNormalization( 256)
			self.bn3_2 = L.BatchNormalization( 256)
			self.bn4_1 = L.BatchNormalization( 512)
			self.bn4_2 = L.BatchNormalization( 512)
			self.bn5_1 = L.BatchNormalization(1012)
			self.bn6_1 = L.BatchNormalization(1012)
			self.bn6_2 = L.BatchNormalization( 512)
			self.bn7_1 = L.BatchNormalization( 512)
			self.bn7_2 = L.BatchNormalization( 256)
			self.bn8_1 = L.BatchNormalization( 256)
			self.bn8_2 = L.BatchNormalization( 128)
			self.bn9_1 = L.BatchNormalization( 128)
			self.bn9_2 = L.BatchNormalization(  64)
			self.bn9_3 = L.BatchNormalization(  64)

	def __call__(self, x): #x = (batchsize, 3, 360, 480)
		if LRN:
			x = F.local_response_normalization(x) #Needed for preventing from overfitting

		h1_1 = F.relu(self.bn1_1(self.enco1_1(x)))
		h1_2 = F.relu(self.bn1_2(self.enco1_2(h1_1)))
		pool1 = F.max_pooling_2d(h1_2, 2, stride=2, return_indices=False) #(batchsize,  64, 180, 240)

		h2_1 = F.relu(self.bn2_1(self.enco2_1(pool1)))
		h2_2 = F.relu(self.bn2_2(self.enco2_2(h2_1)))
		pool2 = F.max_pooling_2d(h2_2, 2, stride=2, return_indices=False) #(batchsize, 128,  90, 120) 

		h3_1 = F.relu(self.bn3_1(self.enco3_1(pool2)))
		h3_2 = F.relu(self.bn3_2(self.enco3_2(h3_1)))
		pool3 = F.max_pooling_2d(h3_2, 2, stride=2, return_indices=False) #(batchsize, 256,  45,  60) 

		h4_1 = F.relu(self.bn4_1(self.enco4_1(pool3)))
		h4_2 = F.relu(self.bn4_2(self.enco4_2(h4_1)))
		pool4 = F.max_pooling_2d(h4_2, 2, stride=2, return_indices=False) #(batchsize, 256,  23,  30) 

		h5_1 = F.relu(self.bn5_1(self.enco5_1(pool4)))

		up5 = F.unpooling_2d(h5_1, ksize=2, stride=2, outsize=(pool3.shape[2], pool3.shape[3]))
		h6_1 = F.relu(self.bn6_1(self.deco6_1(F.concat((up5, h4_2)))))
		h6_2 = F.relu(self.bn6_2(self.deco6_2(h6_1)))

		up6 = F.unpooling_2d(h6_2, ksize=2, stride=2, outsize=(pool2.shape[2], pool2.shape[3]))
		h7_1 = F.relu(self.bn7_1(self.deco7_1(F.concat((up6, h3_2)))))
		h7_2 = F.relu(self.bn7_2(self.deco7_2(h7_1)))

		up7 = F.unpooling_2d(h7_2, ksize=2, stride=2, outsize=(pool1.shape[2], pool1.shape[3]))
		h8_1 = F.relu(self.bn8_1(self.deco8_1(F.concat((up7, h2_2)))))
		h8_2 = F.relu(self.bn8_2(self.deco8_2(h8_1)))

		up8 = F.unpooling_2d(h8_2, ksize=2, stride=2, outsize=(x.shape[2], x.shape[3])) #x = (batchsize, 128, 360, 480)
		h9_1 = F.relu(self.bn9_1(self.deco9_1(F.concat((up8, h1_2)))))
		h9_2 = F.relu(self.bn9_2(self.deco9_2(h9_1)))
		h9_3 = F.relu(self.bn9_3(self.deco9_3(h9_2)))
		
		h = self.final_layer(h9_3)

		return h
