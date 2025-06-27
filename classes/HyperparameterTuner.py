import optuna
from tensorflow.keras import datasets, layers, models
import numpy as np
import tensorflow as tf


class HyperparameterTuner:
    def __init__(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data
        self._configure_gpu()

    def _configure_gpu(self):
        print("Setting up GPUs...")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                print(f'Using GPU: {gpus[0]}')
            except RuntimeError as e:
                print(e)
        else:
            print("No GPUs available, using CPU instead!")


    def objective(self, trial):
        print("Creating new Model...")
        model = models.Sequential()

        num_conv_layers = trial.suggest_int('num_conv_layers', 1, 10)
        num_filters = trial.suggest_categorical('num_filters', [32, 64, 128, 256])
        filter_size = trial.suggest_int('filter_size', 3, 5)
        dense_units = trial.suggest_int('dense_units', 128, 512)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

        model.add(layers.Conv2D(num_filters, (filter_size, filter_size), activation='relu', input_shape=(32, 32, 3),
                                padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Dynamisch Layer hinzufügen
        current_input_shape = (32 // 2, 32 // 2)  # Initial input shape after the first max pooling
        print(f"Adding {num_conv_layers} convolutional layers with {num_filters} filters...")
        for i in range(num_conv_layers - 1):
            model.add(layers.Conv2D(num_filters, (filter_size, filter_size), activation='relu', padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))

            # Calculate new input shape
            current_input_shape = (current_input_shape[0] // 2, current_input_shape[1] // 2)

            # If input shape is too small, stop adding more layers
            if current_input_shape[0] < filter_size or current_input_shape[1] < filter_size:
                print(
                    f"Stopped adding layers (at layer {i + 1}) to prevent negative dimension size. Current input shape: {current_input_shape}")
                break

        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(len(np.unique(self.training_data['Y'])), activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            self.training_data['X'],
            self.training_data['Y'],
            epochs=30,
            batch_size=128,
            validation_data=(self.validation_data['X'], self.validation_data['Y']),
            verbose=1
        )

        loss, accuracy = model.evaluate(self.validation_data['X'], self.validation_data['Y'])
        return accuracy

    def objective_2(self, trial):
        dense_units = trial.suggest_int('dense_units', 450, 1024)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epoch_count = trial.suggest_int('epoch_count', 10, 50)

        print(f"Training model with dense_units={dense_units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}, epoch_count={epoch_count}")

        model = models.Sequential()

        model.add(layers.Input(shape=(32, 32, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))

        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Flatten())

        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(43, activation="softmax"))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

        model.fit(
            self.training_data['X'],
            self.training_data['Y'],
            epochs=epoch_count,
            batch_size=batch_size,
            validation_data=(self.validation_data['X'], self.validation_data['Y']),
            verbose=1,
            callbacks=[early_stop]
        )

        loss, accuracy = model.evaluate(self.validation_data['X'], self.validation_data['Y'])

        if accuracy >= 0.99:
            model.save(f'saved_model/optuna/best/model-{str(int(accuracy * 10000)).replace(".", "")}-{str(np.random.randint(0, 100000))}.h5')

        return accuracy

    def tune(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_2, n_trials=n_trials)
        print('*' * 100)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("Value:", trial.value)
        print("Best Params:")
        print(study.best_params)
        print('*' * 100)

        return study.best_trial