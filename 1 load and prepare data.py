# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:07:23 2019

@author: Ning
"""

#%%
# to install packages directly in Python, not this EDI
# pip install tensorflow
# pip install opencv-python

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from astropy.visualization import make_lupton_rgb # for making own rgb array
#%% set current working directory
os.chdir("D:\\python\\sex data")
#%% read image files 
def load_images_from_folder(folder):
    """
    Reads all the images from a folder and create a list of images
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

men = load_images_from_folder("men")
women = load_images_from_folder('women')

#%% rescale image files to 64*64 pixels, normalize 

def rescale_images(images, num_px = 64):
    """
    Input: a list of images
    Output: an numpy array of images, resized to num_px * num_px size, flattened, normalized,
    of the shape [m, num_px*num_px]
    """
    images_rescaled = []
    for img in images:
        if img is not None:
            img_scaled = np.array(Image.fromarray(img).resize((num_px,num_px)/255))
            img_flattened = img_scaled.reshape((num_px*num_px*3,1))
            images_rescaled.append(img_flattened)
    images_array = np.array(images_rescaled)
    n_x = images_array.shape[1]
    m = images_array.shape[0]
    images_array = images_array.reshape((m, n_x))
    return images_array

men_array = rescale_images(men)
women_array = rescale_images(women)

print(men_array.shape)
print(women_array.shape)



#%% plot the original images
i = 5
plt.imshow(women[i])

#%% plot the rescaled images
num_px = 64
img_reduced = women_array[i,].reshape((num_px, num_px, 3))*255
image = make_lupton_rgb(img_reduced[:,:,2], img_reduced[:,:,1], img_reduced[:,:,0], stretch=0.5)
plt.imshow(image)
#%%
# create labels: 1 as men, 0 as women
m1 = len(men) # sample size for men
m2 = len(women) # sample size for women
m = m1 + m2
n_x = men_array.shape[1]
labels1 = np.ones(m1)
labels0 = np.zeros(m2)
all_labels = np.concatenate([labels1, labels0])
all_labels = all_labels.reshape([m,1])

print("number of male pictures, m1 = " + str(m1))
print("number of female pictures, m2 = " + str(m2))
print("total number of pictures, m = " + str(m))
print("number of input features, n_x = " + str(n_x))

# number of male pictures, m1 = 1410
# number of female pictures, m2 = 1901
# total number of pictures, m = 3311
# number of input features, n_x = 12288

#%% create X, each row is an individual
alldata = np.concatenate([men_array, women_array], axis = 0)
alldata.shape

#%% splitting into training and testing data
train_x, test_x, train_y, test_y = train_test_split(
   alldata, all_labels, test_size=0.15, random_state=726)

print("Number of testing examples: " + str(test_x.shape[0]))
print("test_x shape:" + str(test_x.shape))
print("test_y shape:" + str(test_y.shape))
print("train_x shape:" + str(train_x.shape))
print("train_y shape:" + str(train_y.shape))

# summary of training set
train_counts = pd.Series(train_y.reshape(train_y.shape[0])).value_counts()
print("training set contains {0} men and {1} women".format(
        train_counts[1], train_counts[0]))
# training set contains 1211 men and 1603 women

# summary of testing set
test_counts = pd.Series(test_y.reshape(test_y.shape[0])).value_counts()
print("training set contains {0} men and {1} women".format(
        test_counts[1], test_counts[0]))
# training set contains 199 men and 298 women