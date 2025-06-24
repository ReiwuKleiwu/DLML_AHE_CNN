import tensorflow as tf

class ModelTrainer:
    def __init__(self, model, train_data, valid_data):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

    def train(self, batch_size=128, epochs=100):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

        self.model.fit(
            x=self.train_data['X'],
            y=self.train_data['Y'],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(self.valid_data['X'], self.valid_data['Y']),
            callbacks=[early_stop]
        )
        return self.model