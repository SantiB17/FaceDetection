from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import shutil
import numpy as np


def predict_image(path):
    model = keras.models.load_model("models/mobile_net_v2.h5")
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])
    pred = model.predict(x)
    print(pred)
    if pred[0] > 0.5:
        print('Santi')
    else:
        print('Not Santi')

def copy_to(DIR, LIST, SOURCE):
    for file in LIST:
        if os.path.getsize(os.path.join(SOURCE, file)) == 0:
            continue
        source_path = os.path.join(SOURCE, file)
        dest_path = os.path.join(DIR, file)
        shutil.copyfile(source_path, dest_path)