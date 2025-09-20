#!/usr/bin/env python3
# explore.py

import sys, os, glob, json, logging
from typing import List, Tuple, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt

# ── Silence third-party log spam (Spotipy HTTP errors etc.) ────────────────
for name in ("spotipy","spotipy.client","spotipy.oauth2","urllib3","requests"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
logging.getLogger().setLevel(logging.CRITICAL)

# Local helpers (you already have these files)
from src.spotify_auth import (
    get_spotify_user_client,
    get_spotify_client_credentials,
)

# ───────────────────────────── Data loading & prep ──────────────────────────

def load_history(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    all_records = []
    for fn in files:
        with open(fn, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))
    df = pd.DataFrame(all_records)

    # seconds column (support both schemas)
    if "ms_played" in df.columns:
        df["seconds"] = df["ms_played"] // 1000
    elif "msPlayed" in df.columns:
        df["seconds"] = df["msPlayed"] // 1000
    else:
        df["seconds"] = 0

    # unify common columns
    if "spotify_track_uri" not in df.columns and "trackUri" in df.columns:
        df["spotify_track_uri"] = df["trackUri"]
    if "master_metadata_track_name" not in df.columns and "trackName" in df.columns:
        df["master_metadata_track_name"] = df["trackName"]
    if "master_metadata_album_artist_name" not in df.columns and "artistName" in df.columns:
        df["master_metadata_album_artist_name"] = df["artistName"]

    # unify timestamp
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    elif "endTime" in df.columns:
        df["ts"] = pd.to_datetime(df["endTime"], errors="coerce")
    else:
        df["ts"] = pd.NaT

    return df


def format_duration(sec: int, unit: str) -> str:
    if unit == "sec":
        return f"{sec}s"
    if unit == "min":
        return f"{sec/60:.2f}m"
    if unit == "days":
        return f"{sec/86400:.1f}d"
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


def top_n(df: pd.DataFrame, by: str, unit: str, n: int = 10) -> List[Tuple[str, str]]:
    if df.empty or by not in df.columns:
        return []
    agg = df.groupby(by)["seconds"].sum().sort_values(ascending=False).head(n)
    return [(str(name), format_duration(int(sec), unit)) for name, sec in agg.items()]

# ───────────────────────────── Charts ───────────────────────────────────────

def plot_hourly_chart(df: pd.DataFrame, only_year: Optional[int] = None) -> None:
    if df.empty or df["ts"].isna().all():
        print("⚠️  No timestamps available to draw the hourly chart.")
        return
    df2 = df.dropna(subset=["ts"]).copy()
    if only_year is not None:
        df2 = df2[df2["ts"].dt.year == only_year]
    df2["hour"] = df2["ts"].dt.hour
    hours = (df2.groupby("hour")["seconds"].sum() / 3600.0).reindex(range(24), fill_value=0.0)

    plt.figure(figsize=(12, 5.5))
    plt.plot(hours.index, hours.values, marker="o")
    plt.xticks(range(24))
    title = "Listening Hours by Hour of Day"
    if only_year is not None:
        title += f" — {only_year}"
    plt.title(title)
    plt.xlabel("Hour of Day")
    plt.ylabel("Total Hours (aggregated)")
    for x, y in zip(hours.index, hours.values):
        if y > 0:
            plt.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

# ─────────────── Spotify neighbors (content-based, quiet fallback) ──────────

def find_neighbors_sp(seed_name: str, df: pd.DataFrame, sp_client, n: int = 10, batch_size: int = 100):
    """
    Content-based neighbors using audio features.
    Returns [] on any Spotify issue so caller can fall back silently.
    """
    try:
        if sp_client is None:
            return []
        tracks = (
            df[["spotify_track_uri", "master_metadata_track_name"]]
            .dropna(subset=["spotify_track_uri"])
            .drop_duplicates(subset=["spotify_track_uri"])
        )
        if tracks.empty:
            return []
        matches = tracks.loc[
            tracks["master_metadata_track_name"] == seed_name, "spotify_track_uri"
        ].tolist()
        if not matches:
            return []
        seed_uri = matches[0]
        uris = tracks["spotify_track_uri"].tolist()

        feats = []
        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            try:
                af = sp_client.audio_features(batch)
                if af:
                    feats.extend([f for f in af if f])
            except Exception:
                return []  # swallow API errors quietly

        if not feats:
            return []

        feat_df = pd.DataFrame(feats, index=[f["id"] for f in feats if f and f.get("id")])
        cols = ["danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
        if any(c not in feat_df.columns for c in cols) or feat_df.empty:
            return []
        X = feat_df[cols].fillna(0)
        X.index = ["spotify:track:" + tid for tid in X.index]  # align ids to full URIs

        if seed_uri not in X.index:
            return []

        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(X)
        idx = list(X.index).index(seed_uri)
        sims = list(enumerate(sim[idx]))
        topk = sorted(sims, key=lambda x: x[1], reverse=True)[1:n+1]

        uri2name = dict(zip(tracks["spotify_track_uri"], tracks["master_metadata_track_name"]))
        idx_to_uri = list(X.index)

        out = []
        for i, score in topk:
            uri = idx_to_uri[i]
            name = uri2name.get(uri)
            if name:
                out.append((name, round(float(score), 2)))
        return out
    except Exception:
        return []

# ───────────── Session co-occurrence neighbors (offline, no API needed) ─────

def find_neighbors_data(seed_name: str, df: pd.DataFrame, n: int = 10, session_gap: int = 1800):
    if df.empty or "ts" not in df.columns:
        return []
    df2 = df.dropna(subset=["ts"]).copy().sort_values("ts")
    df2["time_diff"] = df2["ts"].diff().dt.total_seconds().fillna(0)
    df2["session_id"] = (df2["time_diff"] > session_gap).cumsum()
    seed_sessions = df2.loc[df2["master_metadata_track_name"] == seed_name, "session_id"].unique()
    if len(seed_sessions) == 0:
        return []
    rec_counts: Dict[str,int] = {}
    for sid in seed_sessions:
        session_tracks = df2[df2["session_id"] == sid]["master_metadata_track_name"]
        for t in session_tracks:
            if t != seed_name:
                rec_counts[t] = rec_counts.get(t, 0) + 1
    return sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:n]

# ───────────────────────── AI playlist (history-aware) ───────────────────────

def _history_blocks(df: pd.DataFrame, top_k: int = 25) -> Dict[str, List]:
    """Return several history slices to feed into the AI."""
    blocks: Dict[str, List] = {"top_tracks_all": [], "top_artists_all": [], "top_tracks_recent": [], "top_artists_recent": [], "seed_cluster": []}
    if df.empty:
        return blocks

    # Overall top tracks/artists
    top_tracks_all = (
        df.groupby(["master_metadata_track_name","master_metadata_album_artist_name"])["seconds"]
        .sum().sort_values(ascending=False).head(top_k).index.tolist()
    )
    top_artists_all = (
        df.groupby("master_metadata_album_artist_name")["seconds"]
        .sum().sort_values(ascending=False).head(top_k).index.tolist()
    )
    blocks["top_tracks_all"] = [f"{t} — {a}" for (t,a) in top_tracks_all]
    blocks["top_artists_all"] = list(map(str, top_artists_all))

    # Recent year slice (if we have timestamps)
    if "ts" in df.columns and df["ts"].notna().any():
        year = int(df["ts"].dropna().dt.year.max())
        dfy = df[df["ts"].dt.year == year]
        if not dfy.empty:
            top_tracks_recent = (
                dfy.groupby(["master_metadata_track_name","master_metadata_album_artist_name"])["seconds"]
                .sum().sort_values(ascending=False).head(top_k).index.tolist()
            )
            top_artists_recent = (
                dfy.groupby("master_metadata_album_artist_name")["seconds"]
                .sum().sort_values(ascending=False).head(top_k).index.tolist()
            )
            blocks["top_tracks_recent"] = [f"{t} — {a}" for (t,a) in top_tracks_recent]
            blocks["top_artists_recent"] = list(map(str, top_artists_recent))

    # Build a small seed cluster from the top 3 overall tracks via co-occurrence
    seeds = [t for (t, _) in top_tracks_all[:3]]
    co_counts: Dict[str,int] = {}
    for seed in seeds:
        for rec, cnt in find_neighbors_data(seed, df, n=20):
            co_counts[rec] = co_counts.get(rec, 0) + cnt
    # Keep top 20 co-occurred as context
    blocks["seed_cluster"] = [t for (t,_) in sorted(co_counts.items(), key=lambda x:x[1], reverse=True)[:20]]
    return blocks


def _history_summary_for_prompt(df: pd.DataFrame, top_k: int = 25) -> str:
    """Turn history blocks into a compact prompt string."""
    b = _history_blocks(df, top_k=top_k)
    lines = []
    if b["top_tracks_all"]:
        lines.append("Overall favorite tracks: " + "; ".join(b["top_tracks_all"]))
    if b["top_artists_all"]:
        lines.append("Overall favorite artists: " + "; ".join(b["top_artists_all"]))
    if b["top_tracks_recent"]:
        lines.append("Recent year top tracks: " + "; ".join(b["top_tracks_recent"]))
    if b["top_artists_recent"]:
        lines.append("Recent year top artists: " + "; ".join(b["top_artists_recent"]))
    if b["seed_cluster"]:
        lines.append("Tracks that co-occur with favorites in the same sessions (seed cluster): " + "; ".join(b["seed_cluster"]))
    return "\n".join(lines)


def generate_ai_playlist(df: pd.DataFrame, vibe: str) -> Optional[List[Tuple[str, str]]]:
    """Use OpenAI with a history-heavy prompt to produce 30 songs."""
    try:
        from src.ai_client import get_openai_client
    except Exception:
        print("⚠️  OpenAI client not available (src/ai_client.py). Skipping AI playlist.")
        return None

    try:
        client = get_openai_client()
    except Exception:
        print("⚠️  OPENAI_API_KEY not configured. Skipping AI playlist.")
        return None

    context = _history_summary_for_prompt(df, top_k=25)

    system = (
        "You are a meticulous music curator. Create a 30-track playlist tailored to the user's real listening history.\n"
        "Rules:\n"
        "1) Use the history context heavily: lean into recurring artists, adjacent scenes, and co-occurring tracks.\n"
        "2) Match the requested vibe precisely (tempo/energy/mood/era as needed).\n"
        "3) Balance comfort & discovery: include ~30–60% familiar adjacent picks, ~40–70% novel but compatible picks.\n"
        "4) Prefer studio versions and widely available tracks. Avoid duplicates.\n"
        "5) Return exactly 30 lines as 'Title — Artist' (em dash). No numbering, no extra text."
    )

    user = f"Vibe: {vibe}\n\nLISTENING HISTORY CONTEXT:\n{context}\n\nNow produce the 30-track list."

    text = None
    # Try Responses API, then Chat Completions
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role":"system","content":[{"type":"text","text":system}]},
                {"role":"user","content":[{"type":"text","text":user}]},
            ],
        )
        text = resp.output_text
    except Exception:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.7,
            )
            text = resp.choices[0].message.content
        except Exception:
            print("⚠️  Failed to call OpenAI. Skipping AI playlist.")
            return None

    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[Tuple[str, str]] = []
    for ln in lines:
        ln = ln.lstrip("•-0123456789. ").strip()
        sep = "—" if "—" in ln else "-"
        if sep in ln:
            title, artist = [p.strip() for p in ln.split(sep, 1)]
            if title and artist:
                items.append((title, artist))
        if len(items) >= 30:
            break

    return items[:30] if items else None

# ─────────────────────────────── Main ───────────────────────────────────────

def main():
    DATA_PATTERN = os.path.join("data", "Streaming_History*.json")

    mode = input(
        "Select input method:\n"
        "  data  - use streaming history JSON files in './data/' (no login)\n"
        "  login - open browser to 'Login with Spotify' (OAuth)\n"
        "Enter choice (data/login): "
    ).strip().lower()

    sp_user = None
    if mode == "login":
        try:
            sp_user = get_spotify_user_client(scopes=["user-read-email"])
            if sp_user:
                print("✅ Logged in with Spotify.")
            else:
                print("❌ Spotify login failed; proceeding without Spotify features.")
        except Exception as e:
            print(f"❌ Spotify login failed: {e}\nProceeding without Spotify features.")

    # App-only client (for audio features) — preferred for content-based recs
    try:
        sp_app = get_spotify_client_credentials()
    except Exception:
        sp_app = None

    files = glob.glob(DATA_PATTERN)
    if not files:
        input(
            "❗ No streaming history JSON files found in './data/'.\n"
            "  Please add your listening history there, then press Enter..."
        )
        files = glob.glob(DATA_PATTERN)
        if not files:
            print("❌ Still no data files found. Exiting.")
            sys.exit(1)

    df = load_history(DATA_PATTERN)
    if df.empty:
        print("❌ Could not load any records from your data files.")
        sys.exit(1)

    unit = input("Select unit for play time (sec/min/days/mix): ").strip().lower()
    if unit not in {"sec", "min", "days", "mix"}:
        unit = "mix"

    print("\nTop 10 songs by play time:")
    for name, dur in top_n(df, "master_metadata_track_name", unit):
        print(f"  {name:30} {dur}")

    print("\nTop 10 artists by play time:")
    for name, dur in top_n(df, "master_metadata_album_artist_name", unit):
        print(f"  {name:30} {dur}")

    want_chart = input("\nShow hourly listening chart? (y/n): ").strip().lower()
    if want_chart == "y":
        year_choice = input("Limit to a specific year? Enter year like 2024 or press Enter for all: ").strip()
        year_val = None
        if year_choice:
            try:
                year_val = int(year_choice)
            except ValueError:
                year_val = None
        plot_hourly_chart(df, only_year=year_val)

    seed = input("\nEnter a seed track (exact name): ").strip()

    # Content-based (prefer app token) → quiet fallback to session co-occurrence
    recs_cb = find_neighbors_sp(seed, df, sp_app or sp_user)
    if recs_cb:
        print(f"\nRecommendations for '{seed}' (content-based, audio features):")
        for name, score in recs_cb:
            print(f"  • {name}  (score {score})")
    else:
        recs = find_neighbors_data(seed, df)
        if recs:
            print(f"\nRecommendations for '{seed}' (session co-occurrence):")
            for name, count in recs:
                print(f"  • {name}  (co-occurs in {count} sessions)")
        else:
            print("No recommendations found (seed not present in your history?).")

    # ── History-aware AI playlist (prints only) ─────────────────────────────
    want_ai = input("\nGenerate an AI playlist (30 songs) based on your vibe? (y/n): ").strip().lower()
    if want_ai == "y":
        vibe = input("Describe the playlist vibe (e.g., 'late-night indie chill'): ").strip()
        items = generate_ai_playlist(df, vibe)
        if items:
            print("\nAI Playlist (30 tracks):")
            for i, (title, artist) in enumerate(items, 1):
                print(f"{i:2}. {title} — {artist}")
        else:
            print("⚠️  Could not generate AI playlist (OpenAI not configured or request failed).")

if __name__ == "__main__":
    main()
