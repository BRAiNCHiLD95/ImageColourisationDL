# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:30:58 2018

@author: BkiD
"""
from keras.preprocessing.image import  array_to_img, img_to_array, load_img
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import os
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import random
import tensorflow as tf
import numpy as np
import cv2

image = cv2.imread("test_images/man.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

image = 1.0/255*image

lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

X = lab[:,:,0]
Y = lab[:,:,1:]
length, width, chans = image.shape
Y = Y/128
X = X.reshape(1, length, width, 1)
Y = Y.reshape(1, length, width, 2)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
print(model.summary())

model.fit(x=X, y=Y, batch_size=1, epochs=500)
print(model.evaluate(X,Y, batch_size=1))
output = model.predict(X)
output *= 128

# Output colorizations
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
#lab = cv2.cvtColor(cur, cv2.COLOR_LAB2RGB)
#cv2.imwrite("color.jpg",lab)
imsave("img_result.png", lab2rgb(cur))
imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))