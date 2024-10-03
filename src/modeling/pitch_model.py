import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as numpy

# This will be our class for the model we use in training and predicting


class PitchModel:
    # Initialize with our class_count, this is hard set to 3 (FB, BB, OS) and input dimensions
    def __init__(self, input_dim, class_count=3):
        self.model = self.build_compile_model(input_dim, class_count)

    # Method to build our sequential model
    # Example object creation
    # predictor = PitchModel(input_dim=train_label.shape[1])
    def build_compile_model(self, input_dim, class_count):
        # Model with using relu activation and softmax for percentage outputs
        model = Sequential([
            Dense(64, activation='relu', input_shape(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, actiavation='relu'),
            Dense(class_count, activation='softmax')
        ])
        # We compile and return the model, using Adam optimizer which is standard for classification
        model.compile(optimizers=Adam(learning_rate=.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    # Example train call
    # train_history = predictor.train(train_features, train_labels)
    def train(self, train_features, train_labels, epochs=100, batch_size=32, validation_split=0.2):
        return self.model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    # Example predict call
    # player_prediction = predictor.predict(predict_features)
    def predict(self, predict_features):
        return self.model.predict(predict_features)
