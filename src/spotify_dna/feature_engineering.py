import pandas as pd

def add_play_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'ms_played' to a new 'play_seconds' column (float).
    """
    df = df.copy()
    df['play_seconds'] = df['ms_played'] / 1000.0
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the timestamp 'ts' (datetime64[ns, UTC]), add:
      - 'hour'        : hour of day (0–23)
      - 'weekday'     : full weekday name (e.g. 'Monday')
      - 'date'        : date part only
    """
    df = df.copy()
    df['hour']    = df['ts'].dt.hour
    df['weekday'] = df['ts'].dt.day_name()
    df['date']    = df['ts'].dt.date
    return df

def engineer_features(data_dir: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline combining ingestion and basic feature engineering:
    """
    df = add_play_seconds(data_dir)
    df = extract_time_features(df)
    return df
