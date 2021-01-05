import os
from util.functions import evaluate_model, predict_image, split_data
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def upload():

    # Get the file from post request
    pic = request.files['pic']

    # Save the file to ./frontend_uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
    basepath, 'frontend_uploads', secure_filename(pic.filename))
    pic.save(file_path)

    # Make prediction
    preds = predict_image(file_path, 'mobile_net_v2.h5')

    return preds, 200

# if __name__ == '__main__':
#     app.run()

# source_path = 'C:/Users/perro/Downloads/Not_Santi_backup/Not Santi'
# train_path = 'data/train/Not Santi'
# val_path = 'data/val/Not Santi'
# test_path = 'data/test/Not Santi'
#
# split_data(source_path, train_path, val_path, test_path, 0.8)