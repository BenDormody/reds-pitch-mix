import tensorflow as tf
import numpy as np

# This is due to recent import errors for tensorflow.keras
sequential = tf.keras.models.Sequential
layers = tf.keras.layers
adam = tf.keras.optimizers.Adam
regularizers = tf.keras.regularizers
losses = tf.keras.losses
scheduler = tf.keras.optimizers.schedules

# This will be our class for the model we use in training and predicting


class PitchModel:
    # Initialize with our class_count, this is hard set to 3 (FB, BB, OS) and input dimensions
    def __init__(self, input_dim, class_count=3):
        self.model = self.build_compile_model(input_dim, class_count)

    # Method to build our sequential model
    # Example object creation
    # predictor = PitchModel(input_dim=train_label.shape[1])
    def build_compile_model(self, input_dim, class_count):
        """
        function that builds and compiles a sequential model multi-class classifier
        """
        model = sequential([
            layers.Dense(128, activation='sigmoid', input_shape=(
                input_dim,)),
            layers.Dense(class_count, activation='softmax')
        ])
        # We compile and return the model, using Adam optimizer which is standard for classification

        starting_learning_rate = .01
        lr_schedule = scheduler.ExponentialDecay(
            starting_learning_rate,
            decay_steps=1000,
            decay_rate=.9,
            staircase=True
        )

        model.compile(optimizer=adam(learning_rate=lr_schedule),
                      loss='kullback_leibler_divergence',
                      metrics=['mean_squared_error', 'mean_absolute_error'])
        return model
    # Example train call
    # train_history = predictor.train(train_features, train_labels)

    def train(self, train_features, train_labels, epochs=2000, batch_size=8, validation_split=0.3):
        """
        function that trains sequential multi-class classifier on following inputs:

        train_features: dataframe
        train_labels: dataframe
        """
        return self.model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    # Example predict call
    # player_prediction = predictor.predict(predict_features)
    def predict(self, predict_features):
        return self.model.predict(predict_features)
