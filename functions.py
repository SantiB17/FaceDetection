from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np


def predict_image(path):
    model = keras.models.load_model("my_keras_model.h5")
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])
    pred = model.predict(x)
    print(pred)
    if pred[0] < 0.5:
        print('Santi')
    else:
        print('Not Santi')