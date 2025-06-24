import os
from PIL import Image
import numpy as np
import pickle


from classes.HyperparameterTuner import HyperparameterTuner
from classes.AHERecognizer import AHERecognizer

architectural_heritage_elements_classes_mapping = {
    'altar': 0,
    'apse': 1,
    'bell_tower': 2,
    'column': 3,
    'dome(inner)': 4,
    'dome(outer)': 5,
    'flying_buttress': 6,
    'gargoyle': 7,
    'stained_glass': 8,
    'vault': 9
}

def load_images_from_folder(base_folder):
    images = []
    labels = []

    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            image_path = os.path.join(subdir, file)

            img = Image.open(image_path)
            img_array = np.array(img)

            if img_array.shape != (64, 64, 3):
                print(f"Fehler beim Bild {file}: {img_array.shape}")
                continue

            images.append(img_array)

            label = os.path.basename(subdir)
            labels.append(architectural_heritage_elements_classes_mapping[label])

    return images, labels

if __name__ == '__main__':
    #test_data_images, test_data_labels = load_images_from_folder('./data/train')

    #with open('data/train.p', 'wb') as f:
    #    pickle.dump({"images": test_data_images, "labels": test_data_labels}, f)

    aheRecognizer = AHERecognizer(
         os.path.join("data", "train.p"),
         os.path.join("data", "valid.p"),
         (64, 64, 3),
         10
    )

    print(aheRecognizer.valid_rgb['Y'][1000])

    model = aheRecognizer.train_cnn_rgb()
    path = aheRecognizer.save_model("saved_model/my_model.h5", model)
    aheRecognizer.evaluate_model("saved_model/my_model.h5")

    # model = tf.keras.models.load_model("saved_model/optuna/best/model-9911-36930.h5")
    # prediction = trafficSignRecognizer.predict_image(model, "test_images/IMG_0525.png")
    # print(f'Prediction: {prediction}')

    # print("Testing model using image augmentation...")
    # trafficSignRecognizer.evaluate_with_augmentation("saved_model/optuna/best/model-9911-36930.h5")

    # tuner = HyperparameterTuner(
    #     trafficSignRecognizer.training_rgb,
    #     trafficSignRecognizer.valid_rgb,
    # )
    # best_trial = tuner.tune(n_trials=25)
    # print(f'Beste Hyperparameter: {best_trial.params}')