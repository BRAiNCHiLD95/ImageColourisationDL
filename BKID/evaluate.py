# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:40:28 2018

@author: BkiD
"""

import cv2
import sys
import time
from helpers import *
from network import *
import tensorflow as tf
import keras
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model, load_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def Main():
	parser = argparse.ArgumentParser(description='Automatic Image & Video Colourisation Using Deep Learning')
	parser.add_argument('-v', help='source video file name', type=str)
	parser.add_argument('-i', help='provide source image file name', type=str)
	parser.add_argument('-b', help='batch mode: provide source directory', type=str)
	args = parser.parse_args()

	# Model Checkpointing 1 - loading
	model = load_model('latest.h5')	

	if args.i:
		image = args.i
		print("\n\nColouring image %s.. Please Wait!\n" %image)
		color_me = []
		color_me.append(img_to_array(load_img(image)))
		color_me = np.array(color_me, dtype=float)
		gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
		color_me_embed = create_inception_embedding(gray_me)
		color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
		color_me = color_me.reshape(color_me.shape+(1,))
		# Test model
		output = model.predict([color_me, color_me_embed])
		output = output * 128
		# Output colorizations
		for i in range(len(output)):
			cur = np.zeros((256, 256, 3))
			cur[:,:,0] = color_me[i][:,:,0]
			cur[:,:,1:] = output[i]
			imsave("dataset/result/img_"+str(i)+".png", lab2rgb(cur))      
			imsave("dataset/result/img_"+str(i)+"_gray.png", rgb2gray(lab2rgb(cur)))
		print("\n\nImage Coloured Successfully!")
	
	elif args.v:
		video = args.v
		#video = 'agony.mp4'
		dir_path = os.path.abspath(video)
		fps = vid2frame(video, dir_path)
		fol_path = os.path.splitext(os.path.basename(dir_path))[0]
		dir_path = os.getcwd()+'\\'+fol_path+'\\'
		#time.sleep(0.5)
		i = 1
		check_training_data (dir_path)
		for filename in os.listdir(dir_path):
			resized_im = resize_training_data(dir_path+filename)
			cv2.imwrite('dataset\\video_dataset\\'+filename, resized_im)
			sys.stdout.write("\rResized: %d images." %i)
			i += 1
		print("\n")
		color_me = []
		try:
			if not os.path.exists("dataset\\result\\"+fol_path):
				os.makedirs("dataset\\result\\"+fol_path)
				print("Directory Created\n")
		except OSError:
			print("Error creating directory", fol_path)
		for f in os.listdir('dataset\\video_dataset\\'):
			color_me.append(img_to_array(load_img('dataset\\video_dataset\\'+f)))
		print ("Preparing %d Frames for Colouring... Please Wait!\n" %(i-1))
		color_me = np.array(color_me, dtype=float)
		gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
		color_me_embed = create_inception_embedding(gray_me)
		color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
		color_me = color_me.reshape(color_me.shape+(1,))
		# Test model
		output = model.predict([color_me, color_me_embed])
		output = output * 128
		# Output colorizations
		print ("Colouring %d Frames... Please Wait!\n" %(i-1))
		for i in range(len(output)):
			cur = np.zeros((256, 256, 3))
			cur[:,:,0] = color_me[i][:,:,0]
			cur[:,:,1:] = output[i]
			imsave("dataset/result/"+os.path.splitext(video)[0]+"/"+str(i)+".png", lab2rgb(cur))      
			#imsave("dataset/result/img_"+str(i)+"_gray.png", rgb2gray(lab2rgb(cur)))
		print("\n\nFrames Coloured Successfully!")
		time.sleep(0.5)
		print("\nStitching Video..")
		folder_path = "dataset/result/"+os.path.splitext(video)[0]+"/"
		frame2vid (folder_path, fps)
	
	elif args.b:
		folder = args.b
		print("\n\nFolder Loaded..\n")
		color_me = []
		i = 1
		for filename in os.listdir(folder):
			   color_me.append(img_to_array(load_img(folder+'//'+filename)))
			   i +=1   		
		j = i-1
		print ("Preparing %d Images for Colouring... Please Wait!\n" %j)	           
		color_me = np.array(color_me, dtype=float)
		if (color_me.shape[3] == 1):
			gray_me = color_me
			print("Grayscale Images Found!")
		else:
			gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
		color_me_embed = create_inception_embedding(gray_me)
		color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
		color_me = color_me.reshape(color_me.shape+(1,))
		# Test model
		output = model.predict([color_me, color_me_embed])
		output = output * 128
		# Output colorizations
		for i in range(len(output)):
			cur = np.zeros((256, 256, 3))
			cur[:,:,0] = color_me[i][:,:,0]
			cur[:,:,1:] = output[i]
			imsave("dataset/result/img_"+str(i)+".png", lab2rgb(cur))      
			imsave("dataset/result/img_"+str(i)+"_gray.png", rgb2gray(lab2rgb(cur)))
		print("\n\n %s Images Coloured Successfully!" %j)
		
if __name__ == '__main__':
	Main()
