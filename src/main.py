from data_processing.aggregate_data import process_data
from data_processing.time_series_process import load_and_process_data, generate_predictions
from modeling.pitch_model import PitchModel
from modeling.plotter import plot_history, plot_pitch_type_distribution
from modeling.time_series_model import TimeSeriesPitchModel
import numpy as np
import pandas as pd
from sklearn.utils import class_weight


def aggregated_main():
    """
    This function creates, trains and uses our sequential regression model for predictions
    """
    features, labels, prediction_features = process_data('data/raw/data.csv')
    model = PitchModel(input_dim=features.shape[1])

    history = model.train(train_features=features, train_labels=labels)

    plot_history(history)
    prediction_batter_ids = prediction_features.pop('BATTER_ID')
    prediction_batter_names = prediction_features.pop('PLAYER_NAME')
    predictions = model.predict(prediction_features)
    # Create a DataFrame from the predictions, separating each index of the arrays into its own column
    predictions_df = pd.DataFrame(
        predictions, columns=['PITCH_TYPE_BB', 'PITCH_TYPE_FB', 'PITCH_TYPE_OS'])
    # Create a DataFrame from prediction_batter_ids and prediction_batter_names
    batter_info_df = pd.DataFrame({
        'BATTER_ID': prediction_batter_ids,
        'PLAYER_NAME': prediction_batter_names,
        'GAME_YEAR': 2024
    })

    # Combine the batter info and predictions into a final DataFrame
    final_df = pd.concat([batter_info_df, predictions_df], axis=1)
    # Finally we will save this to a csv
    final_df.to_csv('predictions.csv', index=False)


def pitch_distribution_plot():
    """
    Plots our distribution of results in the predictions.csv file
    """
    df = pd.read_csv('predictions.csv')
    plot_pitch_type_distribution(df)


def time_series_main():
    """
    This is a failed model. It has been left in
    to show a issue with many model types for this problem
    """
    features_train, features_test, labels_train, labels_test, label_encoder, last_sequences = load_and_process_data(
        'data/raw/data.csv', sequence_length=50)
    class_indices = np.argmax(labels_train, axis=1)
    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(
                                                          class_indices),
                                                      y=class_indices)
    # Convert to a dictionary
    class_weight_dict = dict(enumerate(class_weights))

    model = TimeSeriesPitchModel(
        input_length=50, input_features=1, class_count=len(label_encoder.classes_))
    history = model.train(train_features=features_train,
                          train_labels=labels_train, class_weight=class_weight_dict)

    batter_predictions = {}

    for batter, last_sequence in last_sequences.items():
        predictions = generate_predictions(
            model.model, last_sequence)
        batter_predictions[batter] = predictions

    for batter, predictions in batter_predictions.items():
        predicted_pitch_types = label_encoder.inverse_transform(predictions)
        print(
            f"Batter {batter} next 100 predicted pitches: {predicted_pitch_types}")
        unique, counts = np.unique(predicted_pitch_types, return_counts=True)
        print(dict(zip(unique, counts)))


if __name__ == '__main__':
    # Put whichever model or plotting functions you want to run here (aggregated_main() is for running the model)
    aggregated_main()
