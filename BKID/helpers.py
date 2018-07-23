# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:13:02 2018

@author: BaD
"""
# Imports
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import numpy as np
from skimage.transform import resize

# Load inception weights

#print("\n\n\t\t\t*****LOADING Inception-ResNet V2*****\n")
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()
print("\n\n\t\t\t*****Inception-ResNet V2 LOADED*****\n")

# HELPER FUNCTIONS

def resize_training_data (image):
	desired_size = 256
	image_size = os.path.getsize(image)
	if (image_size < 1000):
		os.remove(image)
		print("Pruned: "+image)
	im = cv2.imread(image)
	old_size = im.shape[:2]
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	im = cv2.resize(im, (new_size[1], new_size[0]))
	del_w = desired_size - new_size[1]
	del_h = desired_size - new_size[0]
	top, bottom = del_h//2, del_h-(del_h//2)
	left, right = del_w//2, del_w-(del_w//2)
	color = [0, 0, 0]
	resized_im = cv2.copyMakeBorder(im, top, bottom, left, right, 
							 cv2.BORDER_CONSTANT, value = color)
	return resized_im

def check_training_data (dir_path):
	print("\n\n*****Checking for 0KB files*****\n\n")
	for f in os.listdir(dir_path):
		image_path = os.path.join(dir_path,  f)
		image_size = os.path.getsize(image_path)
		if (image_size >= 1000):
			continue
		else:
			os.remove(image_path)
			print("Pruned: "+image_path)


def create_inception_embedding(grayscaled_rgb):
	grayscaled_rgb_resized = []
	for i in grayscaled_rgb:
		i = resize(i, (299, 299, 3), mode='constant')
		grayscaled_rgb_resized.append(i)
	grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
	grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
	with inception.graph.as_default():
		embed = inception.predict(grayscaled_rgb_resized)
	return embed

# Frame Extraction

def vid2frame(video, dir_path):
	cap = cv2.VideoCapture(video)
	dirname = (os.path.splitext(os.path.basename(dir_path))[0])
	try:
		if not os.path.exists(dirname):
			os.makedirs(dirname)
	except OSError:
		print("Error creating directory", dirname)
	currFrame = 1
	fps = cap.get(cv2.CAP_PROP_FPS)
	ret = True
	dirname = os.getcwd() + '\\' + dirname
	while ret:
		ret, frame = cap.read()
		filename = str(currFrame) + '.png'
		cv2.imwrite(dirname+'\\'+filename, frame)
		currFrame += 1
	cap.release()
	cv2.destroyAllWindows()
	print(dirname, "created and", currFrame - 1, "frames extracted @", fps, "FPS.")
	return fps

# Frame Stitching

def frame2vid (folder_path, fps):
	framerate = int(fps)
	dir_path = folder_path
	images = []
	output = (os.path.basename(os.path.dirname(folder_path))+'.mp4')
	for f in os.listdir(dir_path):
		if f.endswith(('.png', '.jpg')):
			images.append(f)
	images = sorted(images, key = lambda x:int(os.path.splitext(x)[0]))
	image_path = os.path.join(dir_path, images[0])
	frame = cv2.imread(image_path)
	regular_size = os.path.getsize(image_path)
	height, width, channels = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output, fourcc, fps, (width, height))
	for n, image in enumerate(images):
		image_path = os.path.join(dir_path, image)
		image_size = os.path.getsize(image_path)
		if image_size < regular_size / 1.5:
			print("Pruned: "+image)
			continue
		frame = cv2.imread(image_path)
		out.write(frame)
		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			break
	out.release()
	cv2.destroyAllWindows()
	print("The output video is {}".format(output))