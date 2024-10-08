import pandas as pd
import numpy as np
from data_processing.pitch_mappings import PITCH_MAP
from sklearn.preprocessing import LabelEncoder


def process_time_series_data(df, sequence_length=30):
    # We will sort the dataframe by batter_id, game_date and at_bat_number
    df_sorted = df.sort_values(['BATTER_ID', 'GAME_DATE', 'AT_BAT_NUMBER'])
    df_sorted['PITCH_TYPE'] = df_sorted['PITCH_TYPE'].map(
        lambda x: PITCH_MAP.get(x, 'Unknown'))
    df_sorted = df_sorted[df_sorted['PITCH_TYPE'].isin((['FB', 'BB', 'OS']))]
    # One-hot encode pitch_type as integers
    le = LabelEncoder()
    df_sorted['PITCH_TYPE_ENCODED'] = le.fit_transform(df_sorted['PITCH_TYPE'])

    # Get the number of unique pitch types
    n_pitch_types = len(le.classes_)

    # Create sequences for each batter
    sequences = []
    labels = []
    last_sequences = {}

    for batter in df_sorted['BATTER_ID'].unique():
        batter_data = df_sorted[df_sorted['BATTER_ID']
                                == batter]['PITCH_TYPE_ENCODED'].values

        for i in range(len(batter_data) - sequence_length):
            seq = batter_data[i:i+sequence_length]
            label = batter_data[i+sequence_length]
            sequences.append(seq)
            labels.append(label)
        # Store the last sequence for each batter
        last_sequences[batter] = batter_data[-sequence_length:]
    X = np.array(sequences)
    y = np.array(labels)

    y_onehot_encoded = np.eye(n_pitch_types)[y]

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y_onehot_encoded, le, last_sequences


def split_data(X, y, test_size=0.2):
    # We first find the index to split at
    split_idx = int(len(X) * (1 - test_size))

    # Split the data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def load_and_process_data(file_path, sequence_length=30, test_size=0.2):

    # Load the data
    df = pd.read_csv(file_path)

    # Process the data
    X, y, le, last_sequences = process_time_series_data(df, sequence_length)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    return X_train, X_test, y_train, y_test, le, last_sequences


def generate_predictions(model, last_sequences, num_predictions=100):
    predictions = []
    current_sequence = last_sequences.copy()

    for _ in range(num_predictions):
        # Here we reshape the current sequence of predictions we are looking at
        input_seq = current_sequence.reshape(1, len(current_sequence), 1)

        # We make a prediction on the current sequence
        pred = model.predict(input_seq)
        predicted_class = np.argmax(pred[0])
        predictions.append(predicted_class)

        # We update the sequence with the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_class

    return np.array(predictions)
