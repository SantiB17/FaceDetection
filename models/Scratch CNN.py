from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

if os.path.isfile('models/my_keras_model.h5') is False:

    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255.)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/perro/PycharmProjects/cv_proj/data/train',
        target_size=(300,300),
        batch_size=20,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/perro/PycharmProjects/cv_proj/data/val',
        target_size=(300,300),
        batch_size=5,
        class_mode='binary'
    )

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(128, (2,2), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Conv2D(128, (2, 2), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=10,
        epochs=30,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=2,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

else:
    model = keras.models.load_model("C:/Users/perro/PycharmProjects/cv_proj/cv_models/my_keras_model.h5")

test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(
        'C:/Users/perro/PycharmProjects/cv_proj/data/test',
        target_size=(300,300),
        batch_size=5,
        class_mode='binary'
    )

loss = model.evaluate(test_generator, steps=5)
