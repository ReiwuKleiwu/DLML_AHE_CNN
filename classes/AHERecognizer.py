import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from classes.DataLoader import DataLoader
from classes.ModelBuilder import ModelBuilder
from classes.ModelTrainer import ModelTrainer
from classes.ModelEvaluator import ModelEvaluator
import tensorflow as tf

architectural_heritage_elements_classes = {
    0: 'altar',
    1: 'apse',
    2: 'bell_tower',
    3: 'column',
    4: 'dome(inner)',
    5: 'dome(outer)',
    6: 'flying_buttress',
    7: 'gargoyle',
    8: 'stained_glass',
    9: 'vault'
}


class AHERecognizer:
    def __init__(self, training_path, valid_path, input_shape, num_classes):
        self.data_loader = DataLoader(training_path, valid_path)
        self.training_rgb, self.valid_rgb, self.training_greyscale, self.valid_greyscale = self.data_loader.load_data()
        self.model_builder = ModelBuilder(input_shape, num_classes)

    def train_fnn_rgb(self):
        model = self.model_builder.create_fnn()
        trainer = ModelTrainer(model, self.training_rgb, self.valid_rgb)
        return trainer.train()

    def train_cnn_rgb(self):
        model = self.model_builder.create_cnn()
        trainer = ModelTrainer(model, self.training_rgb, self.valid_rgb)
        return trainer.train()

    def train_fnn_greyscale(self):
        model = self.model_builder.create_fnn()
        trainer = ModelTrainer(model, self.training_greyscale, self.valid_greyscale)
        return trainer.train()

    def train_cnn_greyscale(self):
        model = self.model_builder.create_cnn()
        trainer = ModelTrainer(model, self.training_greyscale, self.valid_greyscale)
        return trainer.train()

    def save_model(self, file_path, model):
        model.save(file_path)
        return file_path

    def evaluate_model(self, file_path, greyscale=False):
        eval_model = tf.keras.models.load_model(file_path)
        valid_data = self.valid_greyscale if greyscale else self.valid_rgb
        evaluator = ModelEvaluator(eval_model, valid_data)
        return evaluator.evaluate()

    def predict_image(self, model, image_path):
        image = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(64, 64),
            interpolation='bilinear'
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        input_arr = input_arr.astype('float32') / 255.  # This is VERY important

        prediction = model.predict(input_arr)

        # Print certainty of prediction
        print(f'Prediction certainty: {(prediction[0][np.argmax(prediction)]) * 100:.2f}%')
        prediction = np.argmax(prediction)
        return architectural_heritage_elements_classes[prediction]

    def evaluate_with_augmentation(self, model_path):
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        datagen.fit(self.valid_rgb['X'])
        augmented_validation_data = datagen.flow(self.valid_rgb['X'], self.valid_rgb['Y'], batch_size=32)
        model = tf.keras.models.load_model(model_path)
        # evaluate your model on the augmented validation data
        loss, accuracy = model.evaluate(augmented_validation_data)
        print(f'Loss: {loss}, Accuracy: {accuracy}')