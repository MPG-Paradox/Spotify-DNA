import pandas as pd
from pandas import Timestamp
import pytest

from src.spotify_dna.analytics import (
    top_songs,
    top_artists,
    peak_listening_hours,
    songs_played_together,
    top_song_pairs,
)

@pytest.fixture
def tiny_df():
    return pd.DataFrame([
        {'ts': Timestamp('2025-07-01T00:00:00Z'), 'ms_played': 60000,
         'master_metadata_track_name': 'A', 'master_metadata_album_artist_name': 'X', 'genre': 'rock'},
        {'ts': Timestamp('2025-07-01T00:05:00Z'), 'ms_played': 120000,
         'master_metadata_track_name': 'A', 'master_metadata_album_artist_name': 'X', 'genre': 'rock'},
        {'ts': Timestamp('2025-07-01T02:00:00Z'), 'ms_played': 90000,
         'master_metadata_track_name': 'B', 'master_metadata_album_artist_name': 'Y', 'genre': 'pop'},
    ])

def test_top_songs(tiny_df):
    df = top_songs(tiny_df, n=2)
    assert list(df['master_metadata_track_name']) == ['A', 'B']
    assert list(df['play_seconds']) == [180.0, 90.0]

def test_top_artists(tiny_df):
    df = top_artists(tiny_df, n=2)
    assert list(df['master_metadata_album_artist_name']) == ['X', 'Y']

def test_peak_listening_hours(tiny_df):
    series = peak_listening_hours(tiny_df)
    assert series.iloc[0] == 180.0
    assert series.index[0] == 0

def test_songs_played_together(tiny_df):
    df = songs_played_together(tiny_df, window_seconds=600)
    assert df.loc[0, ['track_a', 'track_b']].tolist() == ['A', 'A']
    assert df.loc[0, 'count'] == 1

def test_top_song_pairs(tiny_df):
    df_pairs = top_song_pairs(tiny_df, n=1, window_seconds=600)
    assert len(df_pairs) == 1
    assert df_pairs.loc[0, 'count'] == 1
    # the pair elements both 'A'
    assert set(df_pairs.loc[0, ['track_a', 'track_b']]) == {'A'}
