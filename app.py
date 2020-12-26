import os
from util.functions import evaluate_model, predict_image, split_data
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

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

# source_path = 'C:/Users/perro/PycharmProjects/web_scraping/kanye'
# train_path = 'data/train/Kanye'
# val_path = 'data/val/Kanye'
# test_path = 'data/test/Kanye'
#
# split_data(source_path, train_path, val_path, test_path, 0.8)