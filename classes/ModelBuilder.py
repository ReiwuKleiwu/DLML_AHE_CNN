from tensorflow.keras import layers, models


class ModelBuilder:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_fnn(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_cnn(self):
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding="same")(inputs)

        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(x)

        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding="same")(x)

        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(10, activation="softmax")(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
