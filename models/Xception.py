from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (299, 299)

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

train_generator = train_datagen.flow_from_directory(
    'C:/Users/perro/PycharmProjects/cv_proj/data/train',
    target_size=IMG_SIZE,
    batch_size=20,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/perro/PycharmProjects/cv_proj/data/val',
    target_size=IMG_SIZE,
    batch_size=10,
    class_mode='categorical'
)
# Create the base model from the pre-trained model Xception
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.Xception(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False
base_model.summary()

checkpoint_cb = keras.callbacks.ModelCheckpoint('Xception.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

preprocess_input = keras.applications.xception.preprocess_input
global_average_layer = keras.layers.GlobalAveragePooling2D()
pred_layer = keras.layers.Dense(3)

inputs = keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = keras.layers.Dropout(0.2)(x)
outputs = pred_layer(x)
model = keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

initial_epochs = 30
history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=validation_generator,
                    callbacks=[checkpoint_cb, early_stopping_cb])

base_model.trainable = True

# Fine tune from this layer onwards
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_generator,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_generator,
                         callbacks=[checkpoint_cb, early_stopping_cb])


