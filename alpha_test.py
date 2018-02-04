'''
cv2.imshow('title', im_name)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import cv2
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential

# load image
image = np.array(cv2.imread('test_images/im1.jpg', 1))
cv2.imshow('title', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Convert to LAB and split L, ab
L_chan = rgb2lab(1.0/255*image)[:,:,0]
ab_chan = rgb2lab(1.0/255*image)[:,:,1:]
ab_chan = ab_chan / 128

#getting image dimensions and reshaping the mapped image
length,width,chans = image.shape
L_chan = L_chan.reshape(1, length, width, 1)
ab_chan = ab_chan.reshape(1, length, width, 2)

#the alpha neural net
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

# Finish model
model.compile(optimizer='rmsprop',loss='mse')

#Train the neural network
model.fit(x=L_chan, y=ab_chan, batch_size=1, epochs=100)

#final output
print(model.evaluate(L_chan, ab_chan, batch_size=1))
output = model.predict(L_chan)
output = output * 128
canvas = np.zeros((length, width, 3))
canvas[:,:,0] = L_chan[0][:,:,0]
canvas[:,:,1:] = output[0]
imsave("img_result.jpg", lab2rgb(canvas))
imsave("img_gray_scale.jpg", rgb2gray(lab2rgb(canvas)))