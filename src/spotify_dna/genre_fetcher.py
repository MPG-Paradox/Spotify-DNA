import os
import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Dict, List

def enrich_with_spotify_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches your listening-history DataFrame with a 'genre' column (list of genres).
    Uses Spotify Client Credentials flow; reads SPOTIPY_CLIENT_ID & _SECRET from env.
    """
    # --- 1) Authenticate via client credentials ---
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET env vars to fetch genres."
        )
    auth = SpotifyClientCredentials()
    sp = Spotify(auth_manager=auth)

    # --- 2) Extract unique track IDs from your URIs ---
    def parse_id(uri: str) -> str:
        # handles "spotify:track:ID" or full URLs
        return uri.split(":")[-1] if uri.startswith("spotify:") else uri.rsplit("/", 1)[-1]

    df = df.copy()
    df["track_id"] = df["spotify_track_uri"].apply(parse_id)
    unique_ids = df["track_id"].dropna().unique().tolist()

    # --- 3) Batch-fetch track → artist IDs ---
    track_to_artists: Dict[str, List[str]] = {}
    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i : i + 50]
        resp = sp.tracks(batch)["tracks"]
        for track in resp:
            track_to_artists[track["id"]] = [a["id"] for a in track["artists"]]

    # --- 4) Batch-fetch artist → genres ---
    all_artist_ids = {aid for aids in track_to_artists.values() for aid in aids}
    artist_to_genres: Dict[str, List[str]] = {}
    artist_ids = list(all_artist_ids)
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i : i + 50]
        resp = sp.artists(batch)["artists"]
        for art in resp:
            artist_to_genres[art["id"]] = art.get("genres", [])

    # --- 5) Build track → genre list (union of its artists) ---
    track_to_genres: Dict[str, List[str]] = {}
    for tid, aids in track_to_artists.items():
        genres = []
        for aid in aids:
            genres.extend(artist_to_genres.get(aid, []))
        track_to_genres[tid] = sorted(set(genres))

    # --- 6) Attach and clean up ---
    df["genre"] = df["track_id"].map(track_to_genres)
    return df.drop(columns=["track_id"])
