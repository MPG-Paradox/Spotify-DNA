import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from .feature_engineering import add_play_seconds, extract_time_features

def top_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top n tracks by total play time (seconds).
    """
    df2 = add_play_seconds(df)
    return (
        df2.groupby('master_metadata_track_name', as_index=False)['play_seconds']
           .sum()
           .sort_values('play_seconds', ascending=False)
           .head(n)
    )

def top_artists(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top n artists by total play time.
    """
    df2 = add_play_seconds(df)
    return (
        df2.groupby('master_metadata_album_artist_name', as_index=False)['play_seconds']
           .sum()
           .sort_values('play_seconds', ascending=False)
           .head(n)
    )

def top_genres(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return top n genres by total play time.
    Assumes df has a 'genre' column.
    """
    df2 = add_play_seconds(df)
    if df2['genre'].dtype == object and df2['genre'].apply(lambda x: isinstance(x, list)).any():
        df2 = df2.explode('genre')
    return (
        df2.groupby('genre', as_index=False)['play_seconds']
           .sum()
           .sort_values('play_seconds', ascending=False)
           .head(n)
    )

def peak_listening_hours(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by hour (0–23) with total play seconds, sorted descending.
    """
    df2 = extract_time_features(add_play_seconds(df))
    return df2.groupby('hour')['play_seconds'].sum().sort_values(ascending=False)

def songs_played_together(df: pd.DataFrame, window_seconds: int = 300) -> pd.DataFrame:
    """
    Identify pairs of tracks listened to within 'window_seconds' of each other.
    Returns DataFrame ['track_a','track_b','count'] sorted by count desc.
    """
    df2 = add_play_seconds(df.sort_values('ts')).reset_index(drop=True)
    pairs = {}
    for i in range(len(df2) - 1):
        a = df2.loc[i, 'master_metadata_track_name']
        b = df2.loc[i+1, 'master_metadata_track_name']
        delta = (df2.loc[i+1, 'ts'] - df2.loc[i, 'ts']).total_seconds()
        if 0 < delta <= window_seconds:
            key = tuple(sorted((a, b)))
            pairs[key] = pairs.get(key, 0) + 1

    records = [
        {'track_a': a, 'track_b': b, 'count': cnt}
        for (a, b), cnt in pairs.items()
    ]
    return (
        pd.DataFrame.from_records(records)
          .sort_values('count', ascending=False)
          .reset_index(drop=True)
    )

def top_song_pairs(
    df: pd.DataFrame,
    n: int = 5,
    window_seconds: int = 300
) -> pd.DataFrame:
    """
    Return the top n pairs of tracks listened to within window_seconds of each other.
    """
    return songs_played_together(df, window_seconds).head(n)

def recommend_similar_tracks(
    df: pd.DataFrame,
    seed_track: str,
    n: int = 3,
    window_seconds: int = 300
) -> List[Tuple[str, int]]:
    """
    Return up to n tracks most frequently played within window_seconds of seed_track.
    Raises ValueError if seed_track not in history.
    """
    # 1) Check seed existence
    if seed_track not in df['master_metadata_track_name'].values:
        raise ValueError(f"No plays of '{seed_track}' found in your history.")

    # 2) Build co-occurrence table
    pairs = songs_played_together(df, window_seconds)

    # 3) Filter for rows involving the seed
    mask = (pairs['track_a'] == seed_track) | (pairs['track_b'] == seed_track)
    sub = pairs[mask].copy()
    if sub.empty:
        return []

    # 4) Identify the “other” track
    sub['other'] = sub.apply(
        lambda r: r['track_b'] if r['track_a'] == seed_track else r['track_a'],
        axis=1
    )

    # 5) Sum counts per other track, sort, take top n
    recs = (
        sub.groupby('other')['count']
           .sum()
           .sort_values(ascending=False)
           .head(n)
    )
    return list(recs.items())

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
