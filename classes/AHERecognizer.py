import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from classes.DataLoader import DataLoader
from classes.HyperparameterTuner import HyperparameterTuner
from classes.ModelBuilder import ModelBuilder
from classes.ModelPredictor import ModelPredictor
from classes.ModelTrainer import ModelTrainer
from classes.ModelEvaluator import ModelEvaluator
from classes.ModelVisualizer import ModelVisualizer


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

    def load_model(self, file_path):
        return tf.keras.models.load_model(file_path)

    def evaluate_model(self, file_path, greyscale=False):
        eval_model = tf.keras.models.load_model(file_path)
        valid_data = self.valid_greyscale if greyscale else self.valid_rgb
        evaluator = ModelEvaluator(eval_model, valid_data)
        return evaluator.evaluate()

    def predict_image(self, model, image_path):
        input_arr = self.data_loader.load_image_as_array(image_path)
        model_predictor = ModelPredictor(model)
        cls, prob = model_predictor.predict(input_arr)

        print(f'Prediction: {cls} ({prob * 100:.2f}%)')

    def visualize_feature_maps(self, model, image_path):
        img_arr = self.data_loader.load_image_as_array(image_path)
        model_visualizer = ModelVisualizer(model)
        model_visualizer.visualize_feature_maps(img_arr)

    def visualize_class_activation_maps(self, model, image_path):
        img_arr = self.data_loader.load_image_as_array(image_path)
        model_visualizer = ModelVisualizer(model)
        model_visualizer.visualize_class_activation_maps(img_arr)

    def tune_model(self, n_trials):
        tuner = HyperparameterTuner(
            self.training_rgb,
            self.valid_rgb,
        )
        return tuner.tune(n_trials)

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