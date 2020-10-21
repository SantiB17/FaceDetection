import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders

IMG_SIZE = (160,160)

# Data preprocessing boilerplate code
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    'C:/Users/perro/PycharmProjects/cv_proj/data/train',
    target_size=IMG_SIZE,
    batch_size=30,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/perro/PycharmProjects/cv_proj/data/val',
    target_size=IMG_SIZE,
    batch_size=10,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/perro/PycharmProjects/cv_proj/data/test',
    target_size=IMG_SIZE,
    batch_size=5,
    class_mode='binary'
    )

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
