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
	"""
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
		h = F.upsampling_2d(h, idx4, 2, stride=2, outsize=(idx3.shape[2], idx3.shape[3]))
	
		h = F.relu(self.bn_deco3_1(self.deco3_1(h)))
		h = F.upsampling_2d(h, idx3, 2, stride=2, outsize=(idx2.shape[2], idx2.shape[3]))

		h = F.relu(self.bn_deco2_1(self.deco2_1(h)))
		h = F.upsampling_2d(h, idx2, 2, stride=2, outsize=(idx1.shape[2], idx1.shape[3]))

		h = F.relu(self.bn_deco1_1(self.deco1_1(h)))
		h = F.upsampling_2d(h, idx1, 2, stride=2, outsize=(x.shape[2], x.shape[3]))

		h = self.deco0_1(h)

		#h = F.reshape(h, (x.shape[0], self.n_class, x.shape[2]*x.shape[3]))

		return h
	"""