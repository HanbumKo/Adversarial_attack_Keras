import keras
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
import matplotlib.pyplot as plt
import os
import sys
import glob
import errno
from PIL import Image
import tensorflow as tf
import warnings
import time
with warnings.catch_warnings():
  import keras # keras is still using some deprectade code
%matplotlib inline

model = inception_v3.InceptionV3(weights='imagenet')

images = np.load("imageNumpy.npy")
labels = np.load("labels.npy")
print(images.shape)
print(labels.shape)

correct = 0
incorrect = 0
for i in range(1000):
  image = images[i].reshape([1, 299, 299, 3])
  predict = int(np.argm0ax(model.predict(image)) + 1)
  label = int(labels[i])
  if predict == label:
    correct += 1
    print("correct", predict, label)
  else:
    incorrect += 1
    print("incorrect ", predict, label)

print("accuracy = ", correct/(correct+incorrect))