#A Keras, Wasserstein Generative Adversarial Network with the generator as a Compositional Pattern Producing Network
#Heavily influenced by http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/
#Novel (maybe?) in that it uses Wasserstein loss to achieve diversity and stability in generating color images


#CPPNs are attractive because they can generate aesthetically pleasing large images without a super-resolution network by inputting fractions of integers during prediction
import tensorflow as tf
from tensorflow.contrib import keras as keras


from keras import backend as K
from keras.layers import TimeDistributed
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,TimeDistributed,Lambda,MaxPooling2D,SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.layers.merge import add
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import plot_model





import numpy as np

import os
import random

import imageio
from scipy import misc, ndimage

#can't change scaler rn and reload. Fix that or at least make a generator mode that only loads generator weights and then fits

sideL = 32#width and height by pixel of resized images
zN = 20#latent vector's size
batch_size = 4#size of batches
load = True#whether to load model from file
lr = .0001#change b1 in adam to .5
batches = 1200
scaler = 4#scales up image

def loadDataset(location):#input string indicating dataset location; output numpy array of images
	files = os.listdir(location)
	os.chdir(location)
	#get only the files we want from the dataset
	remover = []
	for i in files:
		filename, file_extension = os.path.splitext(i)
		if file_extension != '.jpg':
			remover.append(i)
	for i in remover:
		files.remove(i)
		
	X_train = np.zeros((len(files),sideL,sideL,3))
	for i in range(len(files)):
		greader = imageio.get_reader(files[i])
		first = greader.get_data(0)
		first = misc.imresize(first,(sideL,sideL,3))
		first = first/255
		X_train[i] = first
	return X_train
	
#initializers
def my_init(shape, dtype=None):
    return K.random_normal_variable(shape=shape, mean=.0, scale=.5)

def small_init(shape, dtype=None):
    return K.random_normal_variable(shape=shape, mean=.0, scale=.01)

def generator(input_z):
	
	with tf.variable_scope('generator'):
		
		#create generator
		
		inputs = Input(tensor = input_z)

		j = Dense(30,activation='linear')(inputs)
		for i in range(4):#JUST TESTING! CHANGE LATER
			z = BatchNormalization()(j)
			x = Dense(30,kernel_initializer=my_init)(z)
			x = Activation('relu')(x)
			x = Dense(30,kernel_initializer=my_init)(x)
			x = Activation('relu')(x)
			x = Dense(30,activation='tanh')(x)
			
			j = add([j,x]) #residual network

		end = Dense(3,activation='sigmoid',name='gend',kernel_initializer=small_init)(j)

		
		#inny = Input(shape=(sideL*sideL,3+zN),tensor = input_z)
		#xer = TimeDistributed(Dense(sideL*sideL))(end)
		
		
		
		gen = Model(inputs,end)
		
		return end,gen
	
def discriminator(image):#I'VE REPLACED BATCHNORMS WITH LAYERNORMS
	alphaL = .2
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		inny = tf.reshape(image,(batch_size,sideL,sideL,3))
		
		conv1 = tf.layers.Conv2D(filters=32,kernel_size=5,strides=2,padding='SAME')
		conv1 = conv1(inny)
		leaky1 = tf.maximum(alphaL * conv1, conv1)
		
		drop1 = tf.layers.dropout(leaky1)
		
		conv2 = tf.layers.Conv2D(filters=64,kernel_size=5,strides=2,padding='SAME')
		conv2 = conv2(drop1)
		batch_norm2 = tf.contrib.layers.layer_norm(conv2, trainable=True)
		leaky2 = tf.maximum(alphaL * batch_norm2, batch_norm2)
		
		drop2 = tf.layers.dropout(leaky2)
		
		conv3 = tf.layers.Conv2D(filters=128,kernel_size=4,strides=2,padding='SAME')
		conv3=conv3(drop2)
		batch_norm3 = tf.contrib.layers.layer_norm(conv3, trainable=True)
		leaky3 = tf.maximum(alphaL * batch_norm3, batch_norm3)
		
		drop3 = tf.layers.dropout(leaky3)
		
		conv4 = tf.layers.Conv2D(filters=256,kernel_size=5,strides=2,padding='SAME')
		conv4=conv4(drop3)
		batch_norm4 = tf.contrib.layers.layer_norm(conv4, trainable=True)
		leaky4 = tf.maximum(alphaL * batch_norm4, batch_norm4)
		
		drop4 = tf.layers.dropout(leaky4)
		
		
		flatter = tf.contrib.layers.flatten(drop4)#default rate = .5
		
		dense = tf.layers.dense(flatter,1)
		
		#out = tf.sigmoid(dense)
		
		return dense
def makeGan():
	input_real = tf.placeholder(tf.float32, shape=(None, sideL, sideL, 3), name='input_real') 
	
	#input_z = tf.placeholder(tf.float32, (None, zN+3), name='input_z')
	
	input_z = K.placeholder(shape=(None,sideL*sideL,zN+3))
	
	input_scale = K.placeholder(shape=(None,sideL*sideL*scaler*scaler,zN+3))
	
	learning_rate = tf.placeholder(tf.float32, name='learning_rate')

	xer,gen = generator(input_z)
	
	_,_ = generator(input_scale)
	
	#gradient penalty adapted from:
	# @misc{wu2016tensorpack,
		  # title={Tensorpack},
		  # author={Wu, Yuxin and others},
		  # howpublished={\url{https://github.com/tensorpack/}},
		  # year={2016}
		# }
	
	
	d_mod_real = discriminator(input_real)
	
	
	d_mod_fake = discriminator(xer)
	
	
	d_loss_real = tf.reduce_mean(d_mod_real)
	
	d_loss_fake = tf.reduce_mean(d_mod_fake)
	
	g_loss = -tf.reduce_mean(d_mod_fake)
	
	alpha = tf.random_uniform(minval=0.,maxval=1.,shape=(batch_size,1,1,1))
	
	inter = input_real + alpha*(tf.reshape(xer,(batch_size,sideL,sideL,3)) - input_real)
	
	outInter = discriminator(inter)
	
	gradients = tf.gradients(outInter,[inter])[0]
	gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients),[1,2,3]))
	gradient_penalty=tf.reduce_mean(tf.square(gradients-1))
	
	
	d_loss = tf.add(d_loss_fake-d_loss_real, 10*gradient_penalty)
	
	
	
	
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
	g_vars = [var for var in t_vars if var.name.startswith('generator')]
	
	
	
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		d_opt = tf.train.AdamOptimizer(lr,.5,.9).minimize(d_loss, var_list=d_vars)
		g_opt = tf.train.AdamOptimizer(lr,.5,.9).minimize(g_loss, var_list=g_vars)
	
	
	return d_opt,g_opt,d_loss,g_loss,input_real,input_z,gen,input_scale

	
#prepare noise and coordinates for larger image
def initialScaledInput():
	bigBoi = np.ones((sideL**2*scaler**2,3+zN))
	noiser = np.random.normal(0, 1, (zN,))
	for x in range(sideL*scaler):
		for y in range(sideL*scaler):
			r = ((x/scaler-sideL/2)**2+(y/scaler-sideL/2)**2)**(1/2)
			#r = ((x/scaler)**2+(y/scaler)**2)**(1/2)
			lil = [x/scaler,y/scaler,r]
			bigBoi[y+x*sideL*scaler] = np.concatenate((noiser,np.array(lil)))
	return bigBoi
#create new noise input
def newScaledNoise():
	noiser = np.random.normal(0, 1, (zN,))
	initScaled[:,:zN] = noiser
	

initScaled = initialScaledInput()

	
#prepare noise and location in image input

intos = np.ones((batch_size,sideL*sideL,3+zN))
lit = np.ones((sideL*sideL,3+zN))
for q in range(batch_size):
	noiser = np.random.normal(0, 1, (zN,))
	for x in range(sideL):
		for y in range(sideL):
			r = ((x-sideL/2)**2+(y-sideL/2)**2)**(1/2)#distance from center of image
			#r = ((x)**2+(y)**2)**(1/2)
			lil = [x,y,r]
			lit[y+x*sideL] = np.concatenate((noiser,np.array(lil)))
	intos[q] = lit
	
#define new noise input
def newNoise():
	for i in range(batch_size):
		noiser = np.random.normal(0, 1, (zN,))			
		intos[i,:,:zN] = noiser
	
def train(i,saveDir,X_train):
	#consider batch sizes
	
	#would move below for 'jq in range(40)', but testing least stable part
	newNoise()
	#get a batch of random images
	idx = np.random.randint(0, X_train.shape[0], batch_size)
	imgs = X_train[idx]
	batch_z = intos
	_ = sess.run(g_opt, feed_dict={input_real: imgs, input_z: batch_z})
	
	
	
	if i == 0:
		plot_model(gen, to_file=saveDir+'model.png')
		for jq in range(100):
			print(jq,'prepping discriminator')
			newNoise()
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]
			
			batch_z = intos
			_ = sess.run(d_opt, feed_dict={input_real: imgs, input_z: batch_z})
			
	
	
	
	train_loss_g = g_loss.eval({input_z: batch_z, input_real: imgs})
	
	
	for qz in range(5):
		idx = np.random.randint(0, X_train.shape[0], batch_size)
		imgs = X_train[idx]
		newNoise()
		batch_z = intos
		_ = sess.run(d_opt, feed_dict={input_real: imgs, input_z: batch_z})
		
		

	
	train_loss_d = d_loss.eval({input_z: batch_z, input_real: imgs})
	
	
	
	print("Step:"+str(i)+" g "+str(train_loss_g)+" d "+str(train_loss_d))
	
	
	
	if (i+1) % 30 == 0 or i == 0: #save weights and sample every while
		if i != 0:
			savedp = saver.save(sess, saveDir+'wcppn.ckpt')
			print(savedp)
		
		newScaledNoise()
		exampIn = np.expand_dims(initScaled,axis=0)
		sample = sess.run(gen(input_scale),feed_dict={input_scale:exampIn})
			
		out = np.resize(sample,(sideL*scaler,sideL*scaler,3))
		
		out = np.maximum(out,np.zeros(out.shape))
		out = np.minimum(out,np.ones(out.shape))
		
		named = saveDir+str(i)+'scaled'+'image.jpg'
		imageio.imwrite(named, out)
		
		for qtc in range(2):
			
			
			newNoise()
			
			exampIn = intos[:1]
			
			#sample = sess.run(generator(input_z),feed_dict={input_z:exampIn})
			
			sample = sess.run(gen(input_z),feed_dict={input_z:exampIn})
			
			out = np.resize(sample,(sideL,sideL,3))
			
			out = np.maximum(out,np.zeros(out.shape))
			out = np.minimum(out,np.ones(out.shape))
			
			named = saveDir+str(i)+' '+str(qtc)+'image.jpg'
			imageio.imwrite(named, out)
		
		
dataDir = os.path.normpath('C:/Users/Com/Python/sort/immFaceDBCropped/')+'\\'#dataset's path
saveDir = os.path.normpath('C:/Users/Com/Desktop/pic/')+'\\'#load and save weights from; save 3 images per batch here as well


x_ = loadDataset(dataDir)
with tf.Session() as sess:
	K.set_session(sess)
	d_opt,g_opt,d_loss,g_loss,input_real,input_z,gen,input_scale = makeGan()
	sess.run(tf.global_variables_initializer())
	
	
	saver = tf.train.Saver()
			
	if load == True:
		saver.restore(sess, saveDir+'wcppn.ckpt')
	
	for i in range(batches):
		train(i,saveDir,x_)
	
	
	
