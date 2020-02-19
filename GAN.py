

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import keras
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import datetime

from keras import initializers

!pip install gdown

import gdown 
gdown.download("https://drive.google.com/uc?id={0}".format("1-JVnG_wVJR3VgAwi6-Hhu2C-ZAyQ2-_9"),"gt.pickle",quiet = False)
gdown.download("https://drive.google.com/uc?id={0}".format("1-7E0x-UGFjotUH8UJAWruM9Y0rwEzYzV"),"occ.pickle",quiet = False)

!ls

pickle_in = open("occ.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("gt.pickle","rb")
y = pickle.load(pickle_in)

from skimage.transform import resize
x = resize(x, (len(x),64,64,1),anti_aliasing=False)
y= resize(y, (len(y),64,64,1),anti_aliasing=False)

print(x.shape)
print(y.shape)

fig = plt.figure(figsize = (6,6))
fig.add_subplot(1,2,1)
plt.imshow(x[2,:,:,0], cmap = "gray")
fig.add_subplot(1,2,2)
plt.imshow(y[2,:,:,0], cmap = "gray")

def creategen():
  generator = Sequential()
  
  generator.add(Conv2D(64,(5,5),strides = (2,2), input_shape = x.shape[1:], padding = "SAME", kernel_initializer = "random_normal" ))
  generator.add(BatchNormalization())
  generator.add(ReLU())
  generator.add(Dropout(0.3))
  
  generator.add(Conv2D(128,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal" ))
  generator.add(BatchNormalization())
  generator.add(ReLU())
  generator.add(Dropout(0.3))
  
  generator.add(Conv2D(256,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal" ))
  generator.add(BatchNormalization())
  generator.add(ReLU())
  generator.add(Dropout(0.3))
  
  
  
  generator.add(Conv2DTranspose(128,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal", use_bias=False ))
  generator.add(BatchNormalization())
  generator.add(ReLU())
  #generator.add(Dropout(0.3))
  
  generator.add(Conv2DTranspose(64,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal", use_bias=False ))
  generator.add(BatchNormalization())
  generator.add(ReLU())
  #generator.add(Dropout(0.3))
  
  
  generator.add(Conv2DTranspose(1,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal", use_bias=False, activation = "sigmoid" ))
  
  return generator



generator = creategen()

def create_disc():
  discriminator = Sequential()
  
  discriminator.add(Conv2D(64,(5,5),strides = (2,2), input_shape = x.shape[1:], padding = "SAME", kernel_initializer = "random_normal" ))
  discriminator.add(BatchNormalization())
  discriminator.add(ReLU())
  discriminator.add(Dropout(0.3))
  
  discriminator.add(Conv2D(128,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal" ))
  discriminator.add(BatchNormalization())
  discriminator.add(ReLU())
  discriminator.add(Dropout(0.3))
  
  discriminator.add(Conv2D(256,(5,5),strides = (2,2), padding = "SAME", kernel_initializer = "random_normal" ))
  discriminator.add(BatchNormalization())
  discriminator.add(ReLU())
  discriminator.add(Dropout(0.3))
  
  discriminator.add(Flatten())
  discriminator.add(Dense(1, activation = "sigmoid"))
  
  return discriminator


discriminator = create_disc()

opt_disc = Adam(lr=0.00004)
discriminator.trainable = True
discriminator.compile(loss = "binary_crossentropy", optimizer = opt_disc)

opt_gen = Adam(lr=0.00001)
generator.compile(loss = "mean_squared_error", optimizer = opt_disc)

def creategan(generator,discriminator):
  gan = Sequential()
  gan.add(generator)
  
  discriminator.trainable = False
  gan.add(discriminator)
  
  return gan

gan = creategan(generator,discriminator)

opt_gan = Adam(lr=0.00001)
gan.compile(loss = "binary_crossentropy", optimizer = opt_gan)

