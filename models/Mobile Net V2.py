import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (160,160)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

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

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

base_model.trainable = False
base_model.summary()

checkpoint_cb = keras.callbacks.ModelCheckpoint("mobile_net_v2.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

preprocess_input = keras.applications.mobilenet_v2.preprocess_input
global_average_layer = keras.layers.GlobalAveragePooling2D()
pred_layer = keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(160, 160, 3))
# x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = pred_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 20
history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=validation_generator,
                    callbacks=[checkpoint_cb, early_stopping_cb])