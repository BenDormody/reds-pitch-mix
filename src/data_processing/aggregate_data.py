from data_processing.pitch_mappings import PITCH_MAP
import pandas as pd

# Method for processing our data into features and labels


def process_data(file_path: str):
    """
    A function that takes in a csv file path and returns:

    features_df: a data frame of features
    labels_df: a matching data frame of labels
    predictions_df: A data frame of our features from 2023 used to predict 2024
    """

    df = pd.read_csv(file_path)
    df['PITCH_TYPE'] = df['PITCH_TYPE'].map(
        lambda x: PITCH_MAP.get(x, 'Unknown'))

    # This will agregate our data by batter_id each year
    # It also unstacks our pitch types to be a count of each one
    pitch_type_df = df.groupby(['BATTER_ID', 'GAME_YEAR', 'PITCH_TYPE']).size(
    ).unstack(fill_value=0).reset_index()
    # This will sum up the strikes, balls, and hits for each batter in each year
    strikes_df = df[df['TYPE'] == 'S'].groupby(
        ['BATTER_ID', 'GAME_YEAR']).size().reset_index(name='Strikes')
    balls_df = df[df['TYPE'] == 'B'].groupby(
        ['BATTER_ID', 'GAME_YEAR']).size().reset_index(name='Balls')
    hits_df = df[df['TYPE'] == 'X'].groupby(
        ['BATTER_ID', 'GAME_YEAR']).size().reset_index(name='Hits').astype(int)

    # Merge strikes, balls, and hits into pitch_type_df to ensure alignment
    pitch_type_df = pitch_type_df.merge(
        strikes_df, on=['BATTER_ID', 'GAME_YEAR'], how='left')
    pitch_type_df = pitch_type_df.merge(
        balls_df, on=['BATTER_ID', 'GAME_YEAR'], how='left')
    pitch_type_df = pitch_type_df.merge(
        hits_df, on=['BATTER_ID', 'GAME_YEAR'], how='left')
    # Calculate total pitches for hit percentage
    pitch_type_df['Total'] = pitch_type_df['BB'] + \
        pitch_type_df['FB'] + pitch_type_df['OS']

    # Calculate the hit percentage for each pitch type
    bb_hit_df = df[(df['PITCH_TYPE'] == 'BB') & (df['TYPE'] == 'X')].groupby(
        ['BATTER_ID', 'GAME_YEAR']).size().reset_index(name='BB_Hit_Percentage')
    fb_hit_df = df[(df['PITCH_TYPE'] == 'FB') & (df['TYPE'] == 'X')].groupby(
        ['BATTER_ID', 'GAME_YEAR']).size().reset_index(name='FB_Hit_Percentage')
    os_hit_df = df[(df['PITCH_TYPE'] == 'OS') & (df['TYPE'] == 'X')].groupby(
        ['BATTER_ID', 'GAME_YEAR']).size().reset_index(name='OS_Hit_Percentage')

    # We are using merge to make sure indexing isn'y messed up
    pitch_type_df = pitch_type_df.merge(
        bb_hit_df, on=['BATTER_ID', 'GAME_YEAR'], how='left')
    pitch_type_df = pitch_type_df.merge(
        fb_hit_df, on=['BATTER_ID', 'GAME_YEAR'], how='left')
    pitch_type_df = pitch_type_df.merge(
        os_hit_df, on=['BATTER_ID', 'GAME_YEAR'], how='left')
    # Convert earlier counts of each hit/pitch type into percentages
    pitch_type_df['BB_Hit_Percentage'] = pitch_type_df['BB_Hit_Percentage'] / \
        pitch_type_df['BB']
    pitch_type_df['FB_Hit_Percentage'] = pitch_type_df['FB_Hit_Percentage'] / \
        pitch_type_df['BB']
    pitch_type_df['OS_Hit_Percentage'] = pitch_type_df['OS_Hit_Percentage'] / \
        pitch_type_df['BB']
    # Calculate our percentage of each pitch type
    pitch_type_df['BB'] = pitch_type_df['BB'] / pitch_type_df['Total']
    pitch_type_df['FB'] = pitch_type_df['FB'] / pitch_type_df['Total']
    pitch_type_df['OS'] = pitch_type_df['OS'] / pitch_type_df['Total']

    # Drop the Total, Unknown, and ? columns which we no longer need
    pitch_type_df = pitch_type_df.drop(columns=['Total', 'Unknown', '?'])

    # Here we shit our game year, batter ID, and pitch types to create our labels for each row
    pitch_type_df['LABEL_GAME_YEAR'] = pitch_type_df['GAME_YEAR'].shift(
        periods=-1, fill_value=0).astype(int)
    pitch_type_df['LABEL_BATTER_ID'] = pitch_type_df['BATTER_ID'].shift(
        periods=-1, fill_value=0).astype(int)
    pitch_type_df['LABEL_BB'] = pitch_type_df['BB'].shift(
        periods=-1, axis=0)
    pitch_type_df['LABEL_FB'] = pitch_type_df['FB'].shift(
        periods=-1, axis=0)
    pitch_type_df['LABEL_OS'] = pitch_type_df['OS'].shift(
        periods=-1, axis=0)

    # Here we get the last year worth of data out for each batter for prediction
    idx = pitch_type_df.groupby('BATTER_ID')['GAME_YEAR'].idxmax()
    predictions_df = pitch_type_df.loc[idx].reset_index(drop=True).fillna(0)
    pitch_type_df = pitch_type_df[pitch_type_df.notna().all(axis=1)]

    pitch_type_df = pitch_type_df[(pitch_type_df['LABEL_GAME_YEAR'] == pitch_type_df['GAME_YEAR'] + 1)
                                  & (pitch_type_df['LABEL_BATTER_ID'] == pitch_type_df['BATTER_ID'])]

    # Finally we drop columns we don't want in our dataset anymore
    labels_df = pitch_type_df[['LABEL_BB', 'LABEL_FB', 'LABEL_OS']]
    labels_to_drop = ['LABEL_GAME_YEAR', 'LABEL_BATTER_ID',
                      'GAME_YEAR', 'LABEL_BB', 'LABEL_FB', 'LABEL_OS']
    features_df = pitch_type_df.drop(
        columns=labels_to_drop).drop(columns=['BATTER_ID'])
    predictions_df = predictions_df.drop(columns=labels_to_drop)
    name_df = df.drop_duplicates(subset='BATTER_ID', keep='first')
    predictions_df = pd.merge(
        predictions_df, name_df[['BATTER_ID', 'PLAYER_NAME']], on='BATTER_ID', how='left')
    return features_df, labels_df, predictions_df
