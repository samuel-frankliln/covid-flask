# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:23:20 2021

@author: samue
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
 
from keras.preprocessing import image

# 

train_datagen =image.ImageDataGenerator(
    rescale =1./255,
    shear_range =0.2,
    zoom_range =0.2,
    horizontal_flip =True,
)
test_dataset = image.ImageDataGenerator(
  rescale =1./255
)

pic_dataset = image.ImageDataGenerator(
  rescale =1./255
)


train_generator =train_datagen.flow_from_directory(
    'C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/Dataset/Train',
    target_size =(224,224),
    batch_size =32,
    class_mode ='binary'
)


train_generator.class_indices





validation_generator =test_dataset.flow_from_directory(
    'C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/Dataset/Val',
    target_size =(224,224),
    batch_size =32,
    class_mode ='binary'

)



pic_generator =pic_dataset.flow_from_directory(
    'C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/program/images',
    target_size =(224,224),
    batch_size =32,
    class_mode ='binary'

)


pretrained_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
pretrained_model.summary()


# Freezing the layers
for layer in pretrained_model.layers[:15]:
    layer.trainable = False
for layer in pretrained_model.layers[15:]:
    layer.trainable = True
 
# Modification of pretrained model
last_layer = pretrained_model.get_layer('block5_pool')
last_output = last_layer.output
 
x = GlobalMaxPooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)
 
# Creating a new model
model = Model(pretrained_model.input, x)
 
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
 
model.summary()
 
# Training model
n_training_samples = len('C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/Dataset/Train')
n_validation_samples = len('C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/Dataset/Val')
 
history = model.fit_generator(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=n_validation_samples//10,
    steps_per_epoch=n_training_samples//10)
 
# Preparing test data
test_dir ='C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/Dataset/Train'
test_images = os.listdir('C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/Dataset/Train')
test_df = pd.DataFrame({
    'image': test_images
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
 

 
# Testing Model
predict = model.predict_generator(validation_generator, steps=np.ceil(nb_samples/10))
threshold = 0.5
test_df['category'] = np.where(predict > threshold, 1,0)