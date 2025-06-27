import numpy as np
from tensorflow.keras.preprocessing import image

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

class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, img_arr):
        prediction = self.model.predict(img_arr)

        cld_id = np.argmax(prediction)
        cls = architectural_heritage_elements_classes[cld_id]
        prob = prediction[0][np.argmax(prediction)]
        return cls, prob