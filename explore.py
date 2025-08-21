#!/usr/bin/env python3
import sys, os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from requests.exceptions import RequestException
from sklearn.metrics.pairwise import cosine_similarity

# ── SPOTIFY CLIENT (set only if creds-mode and creds are valid) ──────────────
sp = None
sp_ok = False  # whether Spotify creds are valid

# ── HELPERS ──────────────────────────────────────────────────────────────────
def pick(df, names):
    """Return the first column name from `names` that exists in df, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None

# ── LOAD & NORMALIZE STREAMING HISTORY ───────────────────────────────────────
def load_history(pattern):
    """
    Loads all JSON files matching pattern and normalizes columns:
      track -> track name
      artist -> artist name
      uri -> spotify track uri/id if present
      ts -> timestamp as datetime
      seconds -> integer seconds of play
    Works with both new ('ts','ms_played',...) and old ('endTime','msPlayed',...) exports.
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")

    records = []
    for fn in files:
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            records.extend(data)

    if not records:
        raise ValueError("No records found in the JSON files.")

    df = pd.DataFrame(records)

    # map possible column names
    ms_col     = pick(df, ["ms_played", "msPlayed"])
    ts_col     = pick(df, ["ts", "endTime"])
    track_col  = pick(df, ["master_metadata_track_name", "trackName"])
    artist_col = pick(df, ["master_metadata_album_artist_name", "artistName"])
    uri_col    = pick(df, ["spotify_track_uri", "trackUri"])

    # seconds
    if ms_col is None:
        raise KeyError("Could not find a playtime column ('ms_played' or 'msPlayed').")
    df["seconds"] = (pd.to_numeric(df[ms_col], errors="coerce").fillna(0) // 1000).astype("int64")

    # timestamp
    if ts_col is None:
        df["ts"] = pd.NaT
    else:
        df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")

    # names
    df["track"] = df[track_col].fillna("Unknown Track") if track_col else "Unknown Track"
    df["artist"] = df[artist_col].fillna("Unknown Artist") if artist_col else "Unknown Artist"

    # uri
    df["uri"] = df[uri_col] if uri_col else pd.NA

    return df

# ── FORMAT DURATION ──────────────────────────────────────────────────────────
def format_duration(sec, unit):
    sec = int(sec)
    if unit == "sec":
        return f"{sec}s"
    if unit == "min":
        return f"{sec/60:.2f}m"
    if unit == "days":
        return f"{sec//86400}d"
    # mix
    d = sec // 86400
    h = (sec % 86400) // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s: parts.append(f"{s}s")
    return " ".join(parts) if parts else "0s"

# ── TOP-N AGGREGATION ────────────────────────────────────────────────────────
def top_n(df, by, unit, n=10):
    agg = df.groupby(by)["seconds"].sum().sort_values(ascending=False).head(n)
    return [(name, format_duration(sec, unit)) for name, sec in agg.items()]

# ── CRED CHECK ───────────────────────────────────────────────────────────────
def check_spotify_creds(client_id, client_secret):
    """
    Try to obtain a token & make a tiny request. Returns (sp_client, ok_bool).
    """
    try:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        client = Spotify(client_credentials_manager=auth, requests_timeout=10, retries=0)
        client.search("test", limit=1)  # tiny call validates token
        return client, True
    except (SpotifyException, RequestException, Exception):
        return None, False

# ── SPOTIFY AUDIO-FEATURE NEIGHBORS (needs valid creds) ──────────────────────
def find_neighbors_sp(seed_name, df, n=10):
    if not sp_ok or sp is None:
        return None  # signal "can't use Spotify path"

    tracks = df[["uri", "track"]].dropna(subset=["uri"]).drop_duplicates(subset=["uri"])
    if tracks.empty:
        return None

    uris = tracks["uri"].tolist()
    try:
        feats = sp.audio_features(uris)
    except Exception:
        return None

    feat_rows = [f for f in feats if f]
    if not feat_rows:
        return None

    feat_df = pd.DataFrame(feat_rows).set_index("uri", drop=True)
    cols = ["danceability","energy","loudness","speechiness",
            "acousticness","instrumentalness","liveness","valence","tempo"]
    X = feat_df[cols].fillna(0)

    sim = cosine_similarity(X)
    uri2name = dict(zip(tracks["uri"], tracks["track"]))

    matches = tracks.loc[tracks["track"] == seed_name, "uri"].tolist()
    if not matches:
        print(f"❌  '{seed_name}' not found in your history.")
        return []

    seed_uri = matches[0]
    if seed_uri not in X.index:
        print("⚠️ Spotify didn’t return audio features for this seed; falling back to data mode.")
        return None

    idx = list(X.index).index(seed_uri)
    sims = list(enumerate(sim[idx]))
    topk = sorted(sims, key=lambda x: x[1], reverse=True)[1:n+1]
    idx2uri = list(X.index)
    return [(uri2name[idx2uri[i]], round(float(score), 2)) for i, score in topk]

# ── SESSION CO-OCCURRENCE NEIGHBORS (no creds) ───────────────────────────────
def find_neighbors_data(seed_name, df, n=10, session_gap=1800):
    if "ts" not in df.columns or df["ts"].isna().all():
        df2 = df.copy()
        df2["session_id"] = 0
    else:
        df2 = df.copy()
        df2 = df2.sort_values("ts")
        diffs = df2["ts"].diff().dt.total_seconds().fillna(session_gap + 1)
        df2["session_id"] = (diffs > session_gap).cumsum()

    seed_sessions = df2.loc[df2["track"] == seed_name, "session_id"].unique()
    if len(seed_sessions) == 0:
        print(f"❌  '{seed_name}' not found in your history.")
        return []

    rec_counts = {}
    for sid in seed_sessions:
        session_tracks = df2[df2["session_id"] == sid]["track"]
        for t in session_tracks:
            if t != seed_name:
                rec_counts[t] = rec_counts.get(t, 0) + 1

    sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    return sorted_recs

# ── PRETTIER HOUR-OF-DAY CHART (POP-UP) ──────────────────────────────────────
def show_hourly_chart(df):
    """
    Displays a pop-up bar chart of total listening by hour:
      • color gradient (viridis) by intensity
      • top 3 hours highlighted and labeled
      • right axis shows percent of total listening
    """
    if "ts" not in df.columns or df["ts"].isna().all():
        print("ℹ️  No timestamps in your data; skipping the hour-of-day chart.")
        return

    hours_index = df["ts"].dt.hour
    sec_by_hour = df.groupby(hours_index)["seconds"].sum().reindex(range(24), fill_value=0)
    hours = np.arange(24)
    vals_hours = sec_by_hour.values / 3600.0  # convert seconds → hours
    total_hours = float(vals_hours.sum())

    # Color mapping by intensity
    cmap = get_cmap("viridis")
    norm = Normalize(vmin=float(vals_hours.min()), vmax=float(vals_hours.max()) if total_hours > 0 else 1.0)
    colors = cmap(norm(vals_hours))

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=115)
    bars = ax.bar(hours, vals_hours, color=colors, edgecolor="#2f2f2f", linewidth=0.8)

    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours])
    ax.set_title("Listening time by hour of day", pad=12, fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour (00–23)")
    ax.set_ylabel("Total listening (hours)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)

    # Annotate top 3 hours
    if total_hours > 0:
        top_idx = np.argsort(vals_hours)[-3:][::-1]
        for i in top_idx:
            bars[i].set_linewidth(1.5)
            bars[i].set_edgecolor("black")
            ax.text(
                i,
                vals_hours[i] + max(vals_hours.max() * 0.02, 0.03),
                f"{vals_hours[i]:.1f}h",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Colorbar key
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Relative intensity")

    # Right axis with % of total listening
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    yt = ax.get_yticks()
    ax2.set_yticks(yt)
    if total_hours > 0:
        ax2.set_yticklabels([f"{(y/total_hours)*100:.0f}%" for y in yt])
    else:
        ax2.set_yticklabels(["0%"] * len(yt))
    ax2.set_ylabel("Share of daily listening")

    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"⚠️  Couldn’t open a pop-up window ({e}). "
              f"If you’re on a headless environment, run locally or install a GUI backend (e.g., Tkinter).")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    global sp, sp_ok

    # 1) MODE SELECTION
    mode = input(
        "Select input method:\n"
        "  data  - use streaming history JSON files in './data/' (no Spotify creds needed)\n"
        "  creds - enter Spotify client_id & client_secret at runtime\n"
        "Enter choice (data/creds): "
    ).strip().lower()

    # 2) CRED COLLECTION (if chosen)
    if mode == "creds":
        cid = input("Enter your Spotify client_id: ").strip()
        csec = input("Enter your Spotify client_secret: ").strip()
        sp, sp_ok = check_spotify_creds(cid, csec)
        if not sp_ok:
            print("⚠️  Spotify credentials are invalid or unavailable. Falling back to data-only recommendations.")

    # 3) ENSURE DATA FILES
    data_pattern = os.path.join("data", "Streaming_History*.json")
    files = glob.glob(data_pattern)
    if not files:
        input(
            "❗ No streaming history JSON files found in './data/'.\n"
            "   Please add your listening history there, then press Enter..."
        )
        files = glob.glob(data_pattern)
        if not files:
            print("❌ Still no data files found. Exiting.")
            sys.exit(1)

    # 4) LOAD & PREP
    df = load_history(data_pattern)

    # 5) UNIT + SUMMARIES
    unit = input("Select unit for play time (sec/min/days/mix): ").strip().lower()
    print("\nTop 10 songs by play time:")
    for name, dur in top_n(df, "track", unit):
        print(f"  {name:30} {dur}")

    print("\nTop 10 artists by play time:")
    for name, dur in top_n(df, "artist", unit):
        print(f"  {name:30} {dur}")

    # 6) HOUR-OF-DAY CHART (POP-UP, prettier)
    show_hourly_chart(df)

    # 7) RECOMMENDATIONS
    seed = input("\nEnter a seed track (exact name): ").strip()

    recs_sp = find_neighbors_sp(seed, df, n=10)
    if isinstance(recs_sp, list):
        print(f"\nRecommendations for '{seed}' (content-based, audio features):")
        for name, score in recs_sp:
            print(f"  • {name}  (score {score})")
        return

    recs = find_neighbors_data(seed, df, n=10)
    if recs:
        print(f"\nRecommendations for '{seed}' (session co-occurrence):")
        for name, count in recs:
            print(f"  • {name}  (co-occurs in {count} sessions)")

if __name__ == "__main__":
    main()
