import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders

# Data preprocessing boilerplate code
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(300, 300),
    batch_size=30,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    'data/val',
    target_size=(300, 300),
    batch_size=10,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(300,300),
    batch_size=5,
    class_mode='binary'
    )
