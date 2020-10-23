from util.functions import predict_image, split_data

path = 'playing_images/santi.jpg'
predict_image(path)

source_path = 'C:/Users/perro/OneDrive/Desktop/test_not_santi'
train_path = 'file_test/train_temp/Not Santi'
val_path = 'file_test/val_temp/Not Santi'
test_path = 'file_test/test_temp/Not Santi'

split_data(source_path, train_path, val_path, test_path, 0.9)