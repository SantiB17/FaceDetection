from util.functions import predict_image, split_data, evaluate_model

path = 'playing_images/santi.jpg'
predict_image(path,'mobile_net_v2.h5')

evaluate_model('my_keras_model.h5')

# source_path = 'C:/Users/perro/Downloads/Photos (1)'
# train_path = 'data/train/Santi'
# val_path = 'data/val/Santi'
# test_path = 'data/test/Santi'
#
# split_data(source_path, train_path, val_path, test_path, 0.8)