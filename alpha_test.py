import numpy as np
import tensorflow as tf

import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

image = img_to_array(load_img('man.jpg'))
image = np.array(image)
image = 1.0/255*image

labImage = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
labImage

np.set_printoptions(precision=4)
X = labImage[:,:,0]
Y = labImage[:,:,1]
Z = labImage[:,:,2]
Y1 = cv2.merge((Y,Z))
Y1