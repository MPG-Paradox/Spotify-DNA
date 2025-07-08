import pandas as pd
from pathlib import Path

from src.spotify_dna.genre_enrichment import (
    load_genre_mapping,
    enrich_with_genre,
)

def test_load_genre_mapping(tmp_path):
    sample = tmp_path / "mapping.csv"
    sample.write_text(
        "spotify_track_uri,genre\n"
        "uri:A,rock\n"
        "uri:B,pop;electronic\n"
    )
    df_map = load_genre_mapping(sample)
    # simple string genre
    assert df_map.loc[0, 'genre'] == 'rock'
    # semicolon-split list
    assert df_map.loc[1, 'genre'] == ['pop', 'electronic']

def test_enrich_with_genre(tmp_path):
    # sample history DataFrame
    df_hist = pd.DataFrame({
        'spotify_track_uri': ['uri:A', 'uri:B', 'uri:C'],
        'master_metadata_track_name': ['A', 'B', 'C']
    })
    # write mapping CSV
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "spotify_track_uri,genre\n"
        "uri:A,rock\n"
        "uri:B,pop\n"
    )
    df_enriched = enrich_with_genre(df_hist, mapping)
    # A and B get their genres, C is NaN
    assert df_enriched.loc[0, 'genre'] == 'rock'
    assert df_enriched.loc[1, 'genre'] == 'pop'
    assert pd.isna(df_enriched.loc[2, 'genre'])
