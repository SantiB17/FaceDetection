from util.functions import predict_image, evaluate_model

path = 'C:/Users/perro/OneDrive/Pictures/Camera Roll/WIN_20201030_16_08_10_Pro.jpg'
predict_image(path,'xception.h5')

evaluate_model('xception.h5')

# source_path = 'C:/Users/perro/Downloads/Photos (1)'
# train_path = 'data/train/Santi'
# val_path = 'data/val/Santi'
# test_path = 'data/test/Santi'
#
# split_data(source_path, train_path, val_path, test_path, 0.8)