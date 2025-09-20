# src/spotify_auth.py
import os
from typing import List, Optional

from dotenv import load_dotenv
import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

# Load variables from .env if present
load_dotenv()

def _env(name: str) -> Optional[str]:
    val = os.getenv(name)
    return val.strip() if val and val.strip() else None

def get_spotify_client_credentials() -> Optional[Spotify]:
    """
    App-only client (no user login). Good for public endpoints like audio features.
    Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your environment.
    """
    cid = _env("SPOTIFY_CLIENT_ID")
    csec = _env("SPOTIFY_CLIENT_SECRET")
    if not cid or not csec:
        return None
    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=csec)
    return spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_user_client(scopes: Optional[List[str]] = None) -> Optional[Spotify]:
    """
    User login via Authorization Code flow (opens browser, caches token).
    Requires SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
    (and you must also add that redirect URI in your Spotify app settings).
    """
    cid = _env("SPOTIFY_CLIENT_ID")
    csec = _env("SPOTIFY_CLIENT_SECRET")
    redirect = _env("SPOTIFY_REDIRECT_URI")
    if not cid or not csec or not redirect:
        return None

    scope_str = " ".join(scopes) if scopes else None
    auth_manager = SpotifyOAuth(
        client_id=cid,
        client_secret=csec,
        redirect_uri=redirect,
        scope=scope_str,
        open_browser=True,
        cache_path=None  # default .cache-* in cwd
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Trigger token fetch and verify it works
    try:
        _ = sp.current_user()
    except Exception:
        return None

    return sp
