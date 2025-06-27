from tensorflow.keras import layers, models


class ModelBuilder:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_fnn(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_cnn(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding="same"))

        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(10, activation="softmax"))

        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model