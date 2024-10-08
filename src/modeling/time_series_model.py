import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
losses = tf.keras.losses
adam = tf.keras.optimizers.Adam


class TimeSeriesPitchModel:
    """
    This is a model that uses sequenced time series data to guess sequences
    of pitch types. This model had multiple issues so it is not used...
    """

    def __init__(self, input_length=30, input_features=None, class_count=3):
        self.input_length = input_length
        self.input_features = input_features
        self.class_count = class_count
        self.model = self.build_compile_model()

    def build_compile_model(self):
        model = models.Sequential([
            layers.LSTM(64, input_shape=(
                self.input_length, self.input_features)),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(self.class_count, activation='softmax')
        ])

        model.compile(optimizer=adam(learning_rate=.01),
                      loss=losses.CategoricalFocalCrossentropy(2),
                      metrics=['accuracy'])
        return model

    def train(self, train_features, train_labels, class_weight, epochs=5, batch_size=32, validation_split=0.2):
        return self.model.fit(train_features, train_labels, epochs=epochs,
                              batch_size=batch_size, class_weight=class_weight, validation_split=validation_split, verbose=1)

    def predict(self, predict_features):
        return self.model.predict(predict_features)
