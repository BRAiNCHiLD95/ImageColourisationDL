# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:26:35 2018

@author: Rajat
"""
# Imports
from helpers import *
from network import *
import cv2
import sys
import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave


# Get images
dir_path = 'dataset/temp_dataset/'
i = 1
print("\n\nResizing Images to 256x256\n")
time.sleep(0.3)
for filename in os.listdir('dataset/train/'):
	resized_im = resize_training_data('dataset/train/'+filename)
	cv2.imwrite('dataset/temp_dataset/'+filename, resized_im)
	sys.stdout.write("\rResized: %d images." %i)
	i += 1
print("\n")

time.sleep(0.25)
check_training_data(dir_path)
time.sleep(0.25)
print("Dataset ready for Training.")
X = []
i = 0
for filename in os.listdir('dataset/temp_dataset/'):
	if (i < 2000):		
		X.append(img_to_array(load_img('dataset/temp_dataset/'+filename)))
		i = i + 1
X = np.array(X, dtype=float)
print("With", X.shape[0], "images of resolution", X.shape[1],"x", X.shape[2])

# Splitting Training & Validation Sets

Xtrain = 1.0/255*X
split = int(0.90*len(Xtrain))
Xtest = rgb2lab(1.0/255*Xtrain[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*Xtrain[split:])[:,:,:,1:]
Ytest = Ytest / 128
batch_size = 10

# Loading model
print("\n\n\t\t\t*****ENCODER-DECODER TRANSFER LEARNING MODEL*****\n")
time.sleep(1.69)
print(model.summary())

print("\n\nTRAINING\n")      
time.sleep(1)
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
batch_size = 10

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

# Train model
time.sleep(0.69)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit_generator(image_a_b_gen(batch_size), epochs=1, steps_per_epoch=200)  

print("Training Done!")
time.sleep(1.5)
print("Loss:\t\t\tAccuracy:")
print(model.evaluate([Xtest, create_inception_embedding(Xtest)], Ytest, verbose=1, batch_size=batch_size))
