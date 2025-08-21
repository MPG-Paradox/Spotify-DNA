#!/usr/bin/env python3
import sys, os, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyOauthError
from sklearn.metrics.pairwise import cosine_similarity

# ── DEFAULT SPOTIFY CREDS (only used if you choose creds mode and enter none) ──
DEFAULT_CLIENT_ID     = "your_client_id"
DEFAULT_CLIENT_SECRET = "your_client_secret"

sp = None  # only set in creds-mode

# ────────────────────────────── Helpers ──────────────────────────────
def _first_present(cols, frame):
    """Return the first column name from cols that exists in frame.columns, else None."""
    for c in cols:
        if c in frame.columns:
            return c
    return None

def load_history(pattern):
    """
    Load all JSON files matching `pattern` and normalize columns:
      ts -> pandas datetime
      track -> track name (falls back to episode name if needed)
      artist -> artist/show name
      uri -> spotify track URI when present
      seconds -> integer seconds played
    """
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    records = []
    for fn in files:
        with open(fn, "r", encoding="utf-8") as f:
            records.extend(json.load(f))
    df = pd.DataFrame(records)

    # --- timestamps ---
    ts_col = _first_present(["ts", "endTime", "time"], df)
    if ts_col:
        df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        df["ts"] = pd.NaT

    # --- seconds played ---
    ms_col = _first_present(["ms_played", "msPlayed", "duration_ms", "ms"], df)
    if ms_col:
        df["seconds"] = (pd.to_numeric(df[ms_col], errors="coerce").fillna(0) // 1000).astype(int)
    elif "seconds" in df.columns:
        df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce").fillna(0).astype(int)
    else:
        df["seconds"] = 0

    # --- track/artist/uri normalization ---
    track_col  = _first_present(["master_metadata_track_name", "trackName", "episode_name"], df)
    artist_col = _first_present(["master_metadata_album_artist_name", "artistName", "episode_show_name"], df)
    uri_col    = _first_present(["spotify_track_uri", "trackUri", "spotify_episode_uri", "episodeUri"], df)

    df["track"]  = df[track_col]  if track_col  else None
    df["artist"] = df[artist_col] if artist_col else None
    df["uri"]    = df[uri_col]    if uri_col    else None

    # keep only rows with some time played
    df = df[df["seconds"] > 0].copy()
    return df

def format_duration(sec, unit):
    if unit == "sec":
        return f"{int(sec)}s"
    if unit == "min":
        return f"{sec/60:.2f}m"
    if unit == "days":
        return f"{sec/86400:.1f}d"
    if unit == "mix":
        sec = int(sec)
        d = sec // 86400
        h = (sec % 86400) // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        parts = []
        if d: parts.append(f"{d}d")
        if h: parts.append(f"{h}h")
        if m: parts.append(f"{m}m")
        if s or not parts: parts.append(f"{s}s")
        return " ".join(parts)
    # fallback (hours)
    return f"{sec/3600:.1f}h"

def top_n(df, by, unit, n=10):
    agg = df.groupby(by)["seconds"].sum().sort_values(ascending=False).head(n)
    return [(name, format_duration(int(sec), unit)) for name, sec in agg.items()]

def ask_unit(prompt, allowed):
    while True:
        u = input(prompt).strip().lower()
        if u in allowed:
            return u
        print(f"Please pick one of: {', '.join(allowed)}")

# ── Spotify-feature neighbors (only in creds-mode) ───────────────────
def find_neighbors_sp(seed_name, df, n=10):
    if sp is None:
        print("❌ Spotify client is not initialized.")
        return []

    tracks = df[["uri", "track"]].dropna(subset=["uri"]).drop_duplicates(subset=["uri"])
    if tracks.empty:
        print("❌ No Spotify URIs in your data.")
        return []

    uris = tracks["uri"].tolist()
    try:
        feats = sp.audio_features(uris)
    except SpotifyOauthError:
        print("⚠️  Invalid Spotify credentials. Falling back to session co-occurrence.")
        return []

    if not feats:
        print("❌ Couldn’t fetch audio features from Spotify.")
        return []

    cols = ["danceability","energy","loudness","speechiness",
            "acousticness","instrumentalness","liveness","valence","tempo"]

    rows, index = [], []
    for f in feats:
        if not f:
            continue
        rows.append([f.get(c, 0) for c in cols])
        index.append(f.get("uri") or f"spotify:track:{f.get('id')}")

    if not rows:
        print("❌ No usable audio feature rows returned.")
        return []

    X = pd.DataFrame(rows, index=index, columns=cols).fillna(0)
    uri2name = dict(zip(tracks["uri"], tracks["track"]))

    matches = tracks.loc[tracks["track"] == seed_name, "uri"].tolist()
    if not matches:
        print(f"❌  '{seed_name}' not found in your history.")
        return []
    seed_uri = matches[0]

    if seed_uri not in X.index:
        print("❌ Seed URI not present in features.")
        return []

    sim = cosine_similarity(X)
    idx = list(X.index).index(seed_uri)
    sims = list(enumerate(sim[idx]))
    topk = sorted(sims, key=lambda x: x[1], reverse=True)[1:n+1]
    return [(uri2name.get(X.index[i], X.index[i]), round(score, 2)) for i, score in topk]

# ── Session co-occurrence neighbors (no creds) ───────────────────────
def find_neighbors_data(seed_name, df, n=10, session_gap=1800):
    df2 = df.copy()
    df2["ts"] = pd.to_datetime(df2["ts"], errors="coerce")
    df2 = df2.sort_values("ts")
    df2["time_diff"] = df2["ts"].diff().dt.total_seconds().fillna(0)
    df2["session_id"] = (df2["time_diff"] > session_gap).cumsum()

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

    return sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:n]

# ── Hour-of-day chart ────────────────────────────────────────────────
def _convert_seconds(arr_seconds, unit):
    arr_seconds = np.array(arr_seconds, dtype=float)
    if unit == "sec":
        return arr_seconds
    if unit == "min":
        return arr_seconds / 60.0
    if unit == "days":
        return arr_seconds / 86400.0
    return arr_seconds / 3600.0  # fallback to hours

def _fmt_unit_value(v, unit):
    if unit == "sec":
        return f"{int(round(v))}s"
    if unit == "min":
        return f"{v:.1f}m"
    if unit == "days":
        return f"{v:.1f}d"
    return f"{v:.1f}h"

def show_hourly_graph(df, unit="min", year=None):
    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
    if year is not None:
        work = work[work["ts"].dt.year == year]

    by_hour = work.groupby(work["ts"].dt.hour)["seconds"].sum()
    by_hour = by_hour.reindex(range(24), fill_value=0)

    y = _convert_seconds(by_hour.values, unit)
    hours = np.arange(24)
    share = y / (y.sum() if y.sum() > 0 else 1)

    fig, ax = plt.subplots(figsize=(13.5, 6))
    bars = ax.bar(hours, y, edgecolor="black", linewidth=1.2)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=y.min(), vmax=y.max() if y.max() > 0 else 1)
    for h, b in zip(hours, bars):
        b.set_facecolor(cmap(norm(y[h])))

    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours])
    title = "Listening time by hour of day"
    if year is not None:
        title += f" – {year}"
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ylabel = {"sec": "Total listening (seconds)",
              "min": "Total listening (minutes)",
              "days": "Total listening (days)"}[unit]
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Hour (00–23)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if y.max() > 0:
        top_idx = np.argsort(y)[-3:][::-1]
        for i in top_idx:
            ax.text(i, y[i] + (y.max() * 0.01), _fmt_unit_value(y[i], unit),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # tiny colorbar showing relative intensity
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, share.max() if share.max() > 0 else 1))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label("Share of daily listening", rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    global sp

    # 1) MODE
    mode = input(
        "Select input method:\n"
        "  data  - use streaming history JSON files in './data/' (no Spotify creds needed)\n"
        "  creds - enter Spotify client_id & client_secret at runtime\n"
        "Enter choice (data/creds): "
    ).strip().lower()

    # 2) SPOTIFY CLIENT (only for creds-mode)
    if mode == "creds":
        cid = input("Enter your Spotify client_id (or leave empty to skip): ").strip() or DEFAULT_CLIENT_ID
        secret = input("Enter your Spotify client_secret (or leave empty to skip): ").strip() or DEFAULT_CLIENT_SECRET
        if cid and secret and cid != "your_client_id" and secret != "your_client_secret":
            try:
                sp = Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret))
            except Exception:
                sp = None
        else:
            sp = None

    # 3) ENSURE DATA FOLDER HAS JSON FILES
    data_pattern = os.path.join("data", "Streaming_History*.json")
    files = glob.glob(data_pattern)
    if not files:
        input(
            "❗ No streaming history JSON files found in './data/'.\n"
            "  Please add your listening history there, then press Enter..."
        )
        files = glob.glob(data_pattern)
        if not files:
            print("❌ Still no data files found. Exiting.")
            sys.exit(1)

    # 4) LOAD & PREPARE HISTORY
    df = load_history(data_pattern)
    if df.empty:
        print("❌ Loaded 0 rows from your history. Exiting.")
        sys.exit(1)

    # 5) TOP LISTS — with mix supported
    unit_lists = ask_unit("Select unit for play time (sec/min/days/mix): ", {"sec", "min", "days", "mix"})

    print("\nTop 10 songs by play time:")
    for name, dur in top_n(df, "track", unit_lists):
        print(f"  {name:30} {dur}")

    print("\nTop 10 artists by play time:")
    for name, dur in top_n(df, "artist", unit_lists):
        print(f"  {name:30} {dur}")

    # 6) NEIGHBOR RECOMMENDATIONS
    seed = input("\nEnter a seed track (exact name): ").strip()
    if mode == "creds" and sp is not None:
        recs = find_neighbors_sp(seed, df)
        if recs:
            print(f"\nRecommendations for '{seed}' (content-based, audio features):")
            for name, score in recs:
                print(f"  • {name}  (score {score})")
        else:
            # fallback to data-only
            recs2 = find_neighbors_data(seed, df)
            if recs2:
                print(f"\nRecommendations for '{seed}' (session co-occurrence, fallback):")
                for name, count in recs2:
                    print(f"  • {name}  (co-occurs in {count} sessions)")
    else:
        recs = find_neighbors_data(seed, df)
        if recs:
            print(f"\nRecommendations for '{seed}' (session co-occurrence):")
            for name, count in recs:
                print(f"  • {name}  (co-occurs in {count} sessions)")

    # 7) OPTIONAL GRAPH — units limited to sec/min/days
    want_graph = input("\nDo you want to see the hourly listening chart? (y/n): ").strip().lower()
    if want_graph in {"y", "yes"}:
        latest_year = int(df["ts"].dt.year.max())
        year_raw = input(
            f"Filter to a specific year? Enter a year (e.g., {latest_year}) or 'all' for everything "
            f"[default: {latest_year}]: "
        ).strip().lower()
        if year_raw == "" or year_raw == "default":
            year = latest_year
        elif year_raw == "all":
            year = None
        else:
            try:
                year = int(year_raw)
            except ValueError:
                print("Could not parse year; using latest.")
                year = latest_year

        unit_graph = ask_unit("Graph unit (sec/min/days): ", {"sec", "min", "days"})
        show_hourly_graph(df, unit=unit_graph, year=year)

if __name__ == "__main__":
    main()
