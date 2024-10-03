import pandas as pd
from pitch_mappings import PITCH_MAP


# Method for processing our data into features and labels
def process_data(file_path):
    df = pd.read_csv(file_path)
    print(df.shape)
    df['PITCH_TYPE'].map(lambda x: PITCH_MAP(x))
    grouped_df = df.groupby(
        by=['BATTER_ID', 'GAME_YEAR', 'PITCH_TYPE']).first()
