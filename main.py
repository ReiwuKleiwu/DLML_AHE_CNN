import os

from classes.AHERecognizer import AHERecognizer

if __name__ == '__main__':
    aheRecognizer = AHERecognizer(
         os.path.join("data", "train.p"),
         os.path.join("data", "valid.p"),
         (64, 64, 3),
         10
    )

    # model = aheRecognizer.train_cnn_rgb()
    # aheRecognizer.save_model("saved_model/my_model.keras", model)
    # aheRecognizer.evaluate_model("saved_model/my_model.keras")
    #
    # model = aheRecognizer.load_model("saved_model/my_model.keras")
    # aheRecognizer.classification_report(model)
    # aheRecognizer.visualize_class_activation_maps(model, 'dome.png')
    # aheRecognizer.visualize_feature_maps(model, 'dome.png')
    # aheRecognizer.predict_image(model, 'dome.png')

    best_trial = aheRecognizer.tune_model(n_trials=25)
    print(f'Beste Hyperparameter: {best_trial.params}')