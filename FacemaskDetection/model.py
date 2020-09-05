#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:47:45 2020
@author: vishwa
"""

#importing libraries
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Lambda, Input
from tensorflow.keras.models import Sequential,Model
import numpy as np


#Specifying Img size and Data Directories
IMG_SIZE = 224
train_path = '/home/vishwa/Documents/COVID Project/Technocolabs-Intern-Project/Face Mask Dataset/Train'
val_path = '/home/vishwa/Documents/COVID Project/Technocolabs-Intern-Project/Face Mask Dataset/Validation'


#Initialising VGG16
vgg = VGG16(input_shape = [IMG_SIZE,IMG_SIZE,3], weights = 'imagenet', include_top = False )

#Not training existing layers
for layer in vgg.layers:
    layer.trainable = False 


X = Flatten()(vgg.output)
prediction = Dense(2, activation = 'softmax')(X)
model = Model(inputs = vgg.input, outputs = prediction)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 15,
                                   width_shift_range = .15,
                                   height_shift_range = .15,
                                   horizontal_flip = True,
                                   zoom_range = 0.2)

train_set = train_datagen.flow_from_directory(train_path, 
                                              target_size = (IMG_SIZE, IMG_SIZE), 
                                              batch_size = 32, 
                                              class_mode = 'categorical')


test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(val_path, 
                                            target_size = (IMG_SIZE, IMG_SIZE), 
                                            batch_size = 32, class_mode = 'categorical')

r = model.fit_generator(train_set, 
                        validation_data = test_set, 
                        epochs = 5, 
                        steps_per_epoch = int(len(train_set)), 
                        validation_steps = int(len(test_set)))

#Testing with Validation Model
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = '/home/vishwa/Documents/COVID Project/Technocolabs-Intern-Project/Face Mask Dataset/Test'

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)


filenames = test_generator.filenames
nb_samples = len(filenames)


predict = model.predict(test_generator,steps = nb_samples)
y_pred = predict>0.5
y_pred = y_pred[:,0]


import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import load_model
model.save('facefeatures.h5')
