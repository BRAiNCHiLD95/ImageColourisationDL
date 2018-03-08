# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:53:08 2018

@author: BkiD
"""
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import cv2
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
import os

# Get images
X = []
for filename in os.listdir('test_images/Train/'):
	X.append(img_to_array(load_img('test_images/Train/'+filename)))  
	print(filename, ".... loaded") 
X = np.array(X, dtype=object)

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = (1.0/255)*Xtrain

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
#X_batch = np.zeros((2,2,1))
batch_size = 2
def image_a_b_gen(batch_size):
	for batch in datagen.flow(Xtrain, batch_size = batch_size):
		lab_batch = rgb2lab(batch)
		X_batch = lab_batch[:,:,0]
		Y_batch = lab_batch[:,:,1:] / 128
		yield (X_batch.reshape(X_batch.shape+(1,)),Y_batch)
			
# model		
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
print(model.summary())

# Train model      
tensorboard = TensorBoard(log_dir="output/first_run")
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=1, steps_per_epoch=2)

# Save model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")