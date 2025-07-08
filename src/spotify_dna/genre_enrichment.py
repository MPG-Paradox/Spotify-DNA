import pandas as pd
from pathlib import Path
from typing import Union

def load_genre_mapping(mapping_file: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV mapping of spotify_track_uri → genre.
    If a genre cell contains semicolons, split into a list.
    """
    df_map = pd.read_csv(mapping_file, dtype=str)
    if 'genre' in df_map.columns:
        df_map['genre'] = df_map['genre'].apply(
            lambda s: s.split(';') if isinstance(s, str) and ';' in s else s
        )
    return df_map

def enrich_with_genre(
    df: pd.DataFrame,
    mapping_file: Union[str, Path]
) -> pd.DataFrame:
    """
    Left-join the genre mapping onto df by 'spotify_track_uri'.
    Adds a new 'genre' column.
    """
    df_map = load_genre_mapping(mapping_file)
    return df.merge(df_map, on='spotify_track_uri', how='left')
