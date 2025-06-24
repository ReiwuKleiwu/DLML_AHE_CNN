import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix


class ModelEvaluator:
    def __init__(self, model, valid_data):
        self.model = model
        self.valid_data = valid_data

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.valid_data['X'], self.valid_data['Y'])
        print('Test Accuracy: {}'.format(accuracy))

        predicted_classes = np.argmax(self.model.predict(self.valid_data['X']), axis=-1)
        y_true = self.valid_data['Y']

        cm = confusion_matrix(y_true, predicted_classes)
        plt.figure(figsize=(25, 25))
        sns.heatmap(cm, annot=True)
        plt.show()
        return loss, accuracy