## SegNet and U-Net by Chainer5.0.0

Detail:[Qiita](https://qiita.com/physics303/items/3fcb0af825a32f48c42e) written in Japanese

## How to run it?

1. Clone it.
2. You have to download CamVid dataset from https://github.com/alexgkendall/SegNet-Tutorial. Make "dataset" directory and put them into the directory.
3. You can change implementation details in config.py.
4. Run by

		python3 train.py
	
	and "saved_models" directory will be made. 

5. Test by

		python3 test.py
		
	and "predicted_imgs" directory will be made. You can see predicted semantic segmantation images in the directory.  

If you want to use U-Net, you have to change 

	network.SegNet()

to

	network.UNet()
	
in train.py and test.py.

## Result

![a](https://user-images.githubusercontent.com/25736044/56456210-120ae100-63a4-11e9-81bc-08c50dd5c774.png)
