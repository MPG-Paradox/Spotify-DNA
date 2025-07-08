import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from .feature_engineering import add_play_seconds, extract_time_features

def top_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top n tracks by total play time (seconds).
    """
    df = add_play_seconds(df)
    grouped = (df
               .groupby('master_metadata_track_name', as_index=False)
               ['play_seconds']
               .sum()
               .sort_values('play_seconds', ascending=False)
               .head(n))
    return grouped

def top_artists(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top n artists by total play time.
    """
    df = add_play_seconds(df)
    grouped = (df
               .groupby('master_metadata_album_artist_name', as_index=False)
               ['play_seconds']
               .sum()
               .sort_values('play_seconds', ascending=False)
               .head(n))
    return grouped

def top_genres(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top n genres by total play time.
    Assumes df has a 'genre' column (string or list of strings).
    """
    df = add_play_seconds(df)
    # explode if genre is list, else treat as single
    if df['genre'].dtype == object and df['genre'].apply(lambda x: isinstance(x, list)).any():
        exploded = df.explode('genre')
    else:
        exploded = df.copy()
    grouped = (exploded
               .groupby('genre', as_index=False)['play_seconds']
               .sum()
               .sort_values('play_seconds', ascending=False)
               .head(n))
    return grouped

def peak_listening_hours(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by hour (0–23) with total play seconds, sorted descending.
    """
    df = extract_time_features(add_play_seconds(df))
    hours = (df
             .groupby('hour')['play_seconds']
             .sum()
             .sort_values(ascending=False))
    return hours

def songs_played_together(df: pd.DataFrame, window_seconds: int = 300) -> pd.DataFrame:
    """
    Identify pairs of tracks listened to within 'window_seconds' of each other.
    Returns a DataFrame with columns ['track_a', 'track_b', 'count'], sorted by count desc.
    """
    df = df.sort_values('ts').reset_index(drop=True)
    df = add_play_seconds(df)
    pairs = {}
    for i in range(len(df) - 1):
        a, b = df.loc[i, 'master_metadata_track_name'], df.loc[i+1, 'master_metadata_track_name']
        delta = (df.loc[i+1, 'ts'] - df.loc[i, 'ts']).total_seconds()
        if 0 < delta <= window_seconds:
            key = tuple(sorted((a, b)))
            pairs[key] = pairs.get(key, 0) + 1

    records = [
        {'track_a': a, 'track_b': b, 'count': cnt}
        for (a, b), cnt in pairs.items()
    ]
    return (pd.DataFrame.from_records(records)
            .sort_values('count', ascending=False)
            .reset_index(drop=True))

# ----- PLOTTING HELPERS -----

def plot_top_artists(df: pd.DataFrame, n: int = 10) -> plt.Figure:
    data = top_artists(df, n)
    fig, ax = plt.subplots()
    ax.bar(data['master_metadata_album_artist_name'], data['play_seconds'])
    ax.set_title(f"Top {n} Artists by Play Time")
    ax.set_ylabel("Play Seconds")
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return fig

def plot_peak_hours(df: pd.DataFrame) -> plt.Figure:
    series = peak_listening_hours(df)
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, marker='o')
    ax.set_title("Peak Listening Hours")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Play Seconds")
    ax.set_xticks(range(0,24,2))
    fig.tight_layout()
    return fig

# You can add similar plot_* functions for top_songs, top_genres, etc.
