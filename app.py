from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
from functions import predict_image
import os

if os.path.isdir('data/train') is False :
    splitfolders.ratio('data', output='data')

if os.path.isfile('my_keras_model.h5') is False:
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(300,300),
        batch_size=30,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'data/val',
        target_size=(300,300),
        batch_size=5,
        class_mode='binary'
    )

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='softmax')
    ])
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=10,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=2
    )

    model.save("my_keras_model.h5")
else:
    model = keras.models.load_model("my_keras_model.h5")

test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(300,300),
        batch_size=5,
        class_mode='binary'
    )

loss = model.evaluate(test_generator, steps=5)

path = 'data/test/Santi/00100lrPORTRAIT_00100_BURST20191231132249104_COVER.jpg'
predict_image(path)