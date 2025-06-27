import tensorflow as tf
import pickle
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.preprocessing import image


class DataLoader:
    def __init__(self, training_path, valid_path):
        self.training_path = training_path
        self.valid_path = valid_path

    def load_data(self):
        with open(self.training_path, mode='rb') as training_data:
            train = pickle.load(training_data)
        with open(self.valid_path, mode='rb') as validation_data:
            valid = pickle.load(validation_data)

        X_train, y_train = train['images'], train['labels']
        X_valid, y_valid = valid['images'], valid['labels']

        X_train, y_train = shuffle(X_train, y_train)
        X_valid, y_valid = shuffle(X_valid, y_valid)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)

        training_rgb = {'X': X_train / 255, 'Y': y_train}
        valid_rgb = {'X': X_valid / 255, 'Y': y_valid}
        training_greyscale = {'X': tf.image.rgb_to_grayscale(training_rgb['X']), 'Y': y_train}
        valid_greyscale = {'X': tf.image.rgb_to_grayscale(valid_rgb['X']), 'Y': y_valid}

        return training_rgb, valid_rgb, training_greyscale, valid_greyscale

    def load_image_as_array(self, image_path):
        img = image.load_img(
            image_path,
            target_size=(64, 64),
            interpolation='bilinear'
        )
        img_arr = image.img_to_array(img)
        img_arr = img_arr.astype('float32') / 255. # Normalize the input
        img_arr = np.expand_dims(img_arr, axis=0) # Transform image into required batch format
        return img_arr
