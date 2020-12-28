from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random
import numpy as np



def predict_image(img_path, model_name):
    models_dir = 'C:/Users/perro/PycharmProjects/cv_proj/models'
    model_path = os.path.join(models_dir, model_name)
    model = keras.models.load_model(model_path)

    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])

    pred = model.predict(x)
    pred_class = pred.argmax(axis=-1)
    labels = ["Kanye", "Not Santi", "Santi"]
    predicted_label = labels[int(pred_class)]

    print(pred, pred_class)
    return predicted_label


def copy_to(DIR, LIST, SOURCE):
    for file in LIST:
        if os.path.getsize(os.path.join(SOURCE, file)) == 0:
            continue
        source_path = os.path.join(SOURCE, file)
        dest_path = os.path.join(DIR, file)
        shutil.copyfile(source_path, dest_path)


def split_data(SOURCE, TRAIN, VAL, TEST, SPLIT_SIZE):
    li = os.listdir(SOURCE)
    num = len(li)
    if num < 10:
        print("Must have at least 10 images to use this function")
        return

    train_set = random.sample(li, int(num * SPLIT_SIZE))
    li = [file for file in li if file not in train_set]
    val_set = random.sample(li, int(0.5 * len(li)))
    test_set = [file for file in li if file not in val_set]

    copy_to(TRAIN, train_set, SOURCE)
    copy_to(VAL, val_set, SOURCE)
    copy_to(TEST, test_set, SOURCE)


def evaluate_model(model_name):
    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_generator = test_datagen.flow_from_directory(
        'C:/Users/perro/PycharmProjects/cv_proj/data/test',
        target_size=(300,300),
        batch_size=5,
        class_mode='binary'
    )

    models_dir = 'C:/Users/perro/PycharmProjects/cv_proj/models'
    model_path = os.path.join(models_dir, model_name)
    model = keras.models.load_model(model_path)

    print(model.evaluate(test_generator, steps=5))


