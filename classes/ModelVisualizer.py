from matplotlib import pyplot as plt
from tensorflow.keras.models import Model

class ModelVisualizer:
    def __init__(self, model):
        self.model = model

    def visualize_feature_maps(self, img_arr):
        # Create a modified model that outputs the activation of every layer
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.layers[0].input, outputs=layer_outputs)

        activations = activation_model.predict(img_arr)

        layer_activation = activations[0] # The first CNN layer
        num_filters = layer_activation.shape[-1]

        plt.figure(figsize=(15, 15))
        for i in range(min(num_filters, 16)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.show()
