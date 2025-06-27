from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


class ModelVisualizer:
    def __init__(self, model):
        self.model = model

    def visualize_feature_maps(self, img_arr):
        # Create a modified model that outputs the activation of every layer
        layer_outputs = [layer.output for layer in self.model.layers]
        # activation_model = Model(inputs=self.model.layers[0].input, outputs=layer_outputs) # Needed for sequential models
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs) # Use this for functional syntax models

        activations = activation_model.predict(img_arr)

        layer_activation = activations[1] # The first CNN layer
        print(layer_activation.shape)
        num_filters = layer_activation.shape[-1]

        plt.figure(figsize=(15, 15))
        for i in range(min(num_filters, 16)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.show()

    def visualize_class_activation_maps(self, img_arr):
        score = CategoricalScore([5])
        gradcam = Gradcam(self.model, model_modifier=ReplaceToLinear(), clone=True)
        cam = gradcam(score, img_arr, penultimate_layer='conv2d')
        heatmap = cam[0]
        plt.imshow(img_arr[0], cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.colorbar()
        plt.show()
