# Reds Pitch Mix Predictor

This project is designed to predict pitch mix probabilities for batters in baseball. It uses machine learning models to analyze historical pitch data and make predictions for future pitch types.

## Project Structure

- `predictions.csv`: Contains the output predictions from the model.
- `requirements.txt`: Lists all Python dependencies for the project.
- `src/main.py`: The main entry point of the application.
- `src/modeling/`: Contains model-related files:
  - `pitch_model.py`: Defines the PitchModel class for prediction.
  - `plotter.py`: Functions for plotting model results and distributions.
  - `time_series_model.py`: An experimental time series model (currently unused).
- `src/data_processing/`: Contains data processing scripts:
  - `aggregate_data.py`: Functions for processing and aggregating pitch data.
  - `pitch_mappings.py`: Defines mappings for pitch types.
  - `time_series_process.py`: Functions for processing time series data (experimental currently unused).
- `data`: Folder that contains all raw data for model and reports

## Reports

All Reports can be found under the `reports` folder and are as follows

- `Technical_Report.pdf` : A report on the workings of the model for those with a technical background

- `Coaching_Staff_Report.pdf` : A report on the model and its results for those with a non-technical background
