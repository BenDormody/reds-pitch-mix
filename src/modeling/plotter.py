import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Function that plots our loss and 2 metrics per epoch of our training
    """
    # Create a figure that takes up 1/3 the horizontal space
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    # Plot our loss and validation loss over each epoch of the process
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # Since our loss we are using is KL divergence we label it as such
    plt.title('Model Loss (KL Divergence)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Graph our MSE in our second horizontal space
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mean_squared_error'], label='Training MSE')
    plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    # Graph our MAE in our final horizontal space of our figure
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'],
             label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_pitch_type_distribution(df):
    # Ensure the required columns are present in the DataFrame
    required_columns = ['PITCH_TYPE_BB', 'PITCH_TYPE_FB', 'PITCH_TYPE_OS']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(
            f"DataFrame must contain the following columns: {required_columns}")

    # Set up the subplots (3 charts side by side)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Titles for the charts
    titles = ['PITCH_TYPE_BB', 'PITCH_TYPE_FB', 'PITCH_TYPE_OS']

    # Plot each histogram
    for i, pitch_type in enumerate(titles):
        axs[i].hist(df[pitch_type], bins=20, range=(0, 1),
                    color='blue', alpha=0.7, edgecolor='black')
        axs[i].set_title(f'Distribution of {pitch_type}')
        axs[i].set_xlabel('Probability (0-1)')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(0, 1)  # Limit x-axis to the probability range 0-1

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plots
    plt.show()
