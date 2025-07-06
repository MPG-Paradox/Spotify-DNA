import json
import pandas as pd
from pathlib import Path
import pytest
from pandas import DatetimeTZDtype

from src.spotify_dna.ingestion import load_streaming_history

# two records: one with master_metadata_track_name, one without
SAMPLE_RECORDS = [
    {
        "ts": "2022-05-22T12:46:51Z",
        "username": "user1",
        "platform": "Windows",
        "ms_played": 60000,
        "conn_country": "IL",
        "ip_addr_decrypted": "0.0.0.0",
        "user_agent_decrypted": "agent",
        "master_metadata_track_name": "Song A",
        "master_metadata_album_artist_name": "Artist A",
        "master_metadata_album_album_name": "Album A",
        "spotify_track_uri": "uri:track:A",
        "episode_name": None,
        "episode_show_name": None,
        "spotify_episode_uri": None,
        "reason_start": "trackdone",
        "reason_end": "trackdone",
        "shuffle": False,
        "skipped": False,
        "offline": False,
        "incognito_mode": False,
        "offline_timestamp": None
    },
    {
        "ts": "2022-05-22T12:47:51Z",
        "username": "user1",
        "platform": "Windows",
        "ms_played": 0,
        "conn_country": "IL",
        "ip_addr_decrypted": "0.0.0.0",
        "user_agent_decrypted": "agent",
        "master_metadata_track_name": None,
        "master_metadata_album_artist_name": None,
        "master_metadata_album_album_name": None,
        "spotify_track_uri": None,
        "episode_name": None,
        "episode_show_name": None,
        "spotify_episode_uri": None,
        "reason_start": "trackdone",
        "reason_end": "trackdone",
        "shuffle": False,
        "skipped": False,
        "offline": False,
        "incognito_mode": False,
        "offline_timestamp": None
    }
]

def test_load_streaming_history(tmp_path):
    # write sample JSON to temp dir
    sample_file = tmp_path / "Streaming_History_Audio_test.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_RECORDS, f)

    # run loader
    df = load_streaming_history(tmp_path)

    # should be a DataFrame with exactly one row (filtering null track_name)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    # ts column must be datetime with UTC tz
    assert isinstance(df['ts'].dtype, DatetimeTZDtype)

    # essential columns must exist
    for col in [
        'ts',
        'username',
        'platform',
        'ms_played',
        'master_metadata_track_name',
        'master_metadata_album_artist_name',
        'master_metadata_album_album_name'
    ]:
        assert col in df.columns

    # verify the one remaining track name
    assert df.loc[0, 'master_metadata_track_name'] == "Song A"
