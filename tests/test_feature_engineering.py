import pandas as pd
from datetime import datetime, timezone
import pytest

from src.spotify_dna.feature_engineering import (
    add_play_seconds,
    extract_time_features,
    engineer_features,
)

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            'ts': pd.Timestamp('2022-05-22T12:46:51Z'),
            'ms_played': 90000,
        },
        {
            'ts': pd.Timestamp('2022-05-23T00:00:00Z'),
            'ms_played': 0,
        }
    ])

def test_add_play_seconds(sample_df):
    df2 = add_play_seconds(sample_df)
    assert 'play_seconds' in df2.columns
    assert df2.loc[0, 'play_seconds'] == 90.0
    assert df2.loc[1, 'play_seconds'] == 0.0

def test_extract_time_features(sample_df):
    df2 = extract_time_features(sample_df)
    assert all(col in df2.columns for col in ['hour', 'weekday', 'date'])
    # First row: 12:46 UTC on Sunday
    assert df2.loc[0, 'hour'] == 12
    assert df2.loc[0, 'weekday'] == 'Sunday'
    assert df2.loc[0, 'date'] == datetime(2022, 5, 22).date()

def test_engineer_features(sample_df):
    df3 = engineer_features(sample_df)
    # Should include all columns from both steps
    for col in ['play_seconds', 'hour', 'weekday', 'date']:
        assert col in df3.columns
    # And original ts stays intact
    assert 'ts' in df3.columns
