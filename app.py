import os

from tensorflow import keras
from keras.applications.imagenet_utils import decode_predictions

from util.functions import predict_image, evaluate_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

path = 'C:/Users/perro/OneDrive/Pictures/Camera Roll/WIN_20201030_16_08_10_Pro.jpg'
predict_image(path,'xception.h5')

evaluate_model('xception.h5')

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(f.filename)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_image(file_path, 'xception.h5')

        # Process result for user
        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)

# source_path = 'C:/Users/perro/Downloads/Photos (1)'
# train_path = 'data/train/Santi'
# val_path = 'data/val/Santi'
# test_path = 'data/test/Santi'
#
# split_data(source_path, train_path, val_path, test_path, 0.8)