# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:43:08 2018

@author: BkiD
"""

from keras.layers import Conv2D, UpSampling2D, Dropout, Activation, InputLayer
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf

# Loading images
X = []
i = 0
for filename in os.listdir('test_images/Train2/'):
    if (i<1000):
        X.append(img_to_array(load_img('test_images/Train2/'+filename)))
        #os.remove('test_images/Train2/'+filename)
        i = i + 1
X = np.array(X, dtype=float)

# Splitting train and test sets
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128

# Model Structure
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
model.add(Dropout(0.5))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
print(model.summary())

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
		
# Model Checkpointing 1 - loading
model = load_model('aivcudl.h5')

# Train model      
model.fit_generator(image_a_b_gen(batch_size), epochs=5, steps_per_epoch=100)

# Model checkpointing 2 - saving
model.save('aivcudl.h5')

# Evaluation
print("Loss:\t\t\tAccuracy:")
print(model.evaluate(Xtest, Ytest, verbose=0, batch_size=batch_size))

# Batch image testing

color_me = []
for filename in os.listdir('test_images/Test2/'):
        color_me.append(img_to_array(load_img('test_images/Test2/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))
output = model.predict(color_me)
output = output * 128
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))
	
# Single image Testing

color_me = load_img('pic.jpg')
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,0]
color_me = color_me.reshape((1,)+color_me.shape+(1,))
output = model.predict(color_me)
output = output * 128
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("img_result.png", lab2rgb(cur))
    imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))