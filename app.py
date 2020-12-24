import os

from tensorflow import keras
from keras.applications.imagenet_utils import decode_predictions

from util.functions import predict_image, evaluate_model
from flask import Flask, request
from werkzeug.utils import secure_filename

# path = 'C:/Users/perro/OneDrive/Pictures/Camera Roll/WIN_20201030_16_08_10_Pro.jpg'
# predict_image(path,'xception.h5')
#
# evaluate_model('xception.h5')

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        pic = request.files['pic']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(pic.filename))
        pic.save(file_path)

        # Make prediction
        preds = predict_image(file_path, 'xception.h5')

        # # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return preds, 200
    return None

if __name__ == '__main__':
    app.run(debug=True)

# source_path = 'C:/Users/perro/Downloads/Photos (1)'
# train_path = 'data/train/Santi'
# val_path = 'data/val/Santi'
# test_path = 'data/test/Santi'
#
# split_data(source_path, train_path, val_path, test_path, 0.8)