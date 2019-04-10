from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
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

# -- cleverhans --
adv_xs = []
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(config=config)
keras.backend.set_session(sess)
wrap = KerasModelWrapper(model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.2, 'clip_min': 0., 'clip_max': 1.}
x = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
# Generate adversarial examples
adv_x = fgsm.generate(x, **fgsm_params)
adv_x = tf.stop_gradient(adv_x)
# Consider the attack to be constant
sess.run(tf.global_variables_initializer())
for i in range(1000):
    start_time = time.time()
    adv_image = adv_x.eval(session=sess, feed_dict={x: images[i].reshape([1, 299, 299, 3])})
    adv_xs.append(list(adv_image))
    # plt.imshow(np.clip(adv_image.reshape(299, 299, 3), 0., 1.))
    # plt.axis('off')
    # plt.show()

adv_xs = np.asarray(adv_xs)
print(adv_xs.shape)

correct = 0
incorrect = 0
model = inception_v3.InceptionV3(weights='imagenet')
for i in range(1000):
    start_time = time.time()

    preds = int(np.argmax(model.predict(adv_xs[i].reshape([1, 299, 299, 3]))) + 1)
    label = int(labels[i])
    if preds == label:
        correct += 1
        #print("Time : %s seconds" % (time.time() - start_time))
        print(i, ": correct", preds, label)
    else:
        incorrect += 1
        #print("Time : %s seconds" % (time.time() - start_time))
        print(i, ": incorrect", preds, label)

print("Num of correct : ", correct)
print("Num of incorrect : ", incorrect)
print("accuracy = ", correct / (correct + incorrect))
