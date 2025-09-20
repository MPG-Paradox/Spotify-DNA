#!/usr/bin/env python3
import os, glob, json, re, sys, platform, subprocess, shutil
import pandas as pd

# Uses your existing OpenAI client wrapper
from src.ai_client import get_openai_client

DATA_PATTERN = os.path.join("data", "Streaming_History*.json")

# ── Load history (handles old/new Spotify JSON schemas) ──────────────────────
def load_history(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    records = []
    for fn in files:
        with open(fn, "r", encoding="utf-8") as f:
            records.extend(json.load(f))

    df = pd.DataFrame(records)

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    ts_col     = pick(["ts", "endTime"])
    ms_col     = pick(["ms_played", "msPlayed"])
    track_col  = pick(["master_metadata_track_name", "trackName"])
    artist_col = pick(["master_metadata_album_artist_name", "artistName"])

    missing = [name for name, col in [
        ("timestamp", ts_col), ("ms_played", ms_col),
        ("track", track_col), ("artist", artist_col)
    ] if col is None]
    if missing:
        raise KeyError(f"Your data is missing required columns: {', '.join(missing)}")

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    out["seconds"] = (df[ms_col].fillna(0).astype("int64") // 1000)
    out["track"] = df[track_col].astype(str)
    out["artist"] = df[artist_col].astype(str)

    out = out.dropna(subset=["ts"])
    out = out[out["seconds"] > 0]
    return out

# ── Taste context helpers ────────────────────────────────────────────────────
def top_items(series: pd.Series, n=20):
    return [name for name, _ in series.value_counts().head(n).items()]

def build_prompt(vibe: str, top_artists: list[str], top_tracks: list[str]) -> str:
    return f"""
You are a music assistant. Create a cohesive 30-track playlist that matches this vibe:

VIBE: {vibe}

Bias the selection toward the user's listening history below, but you may also
include a few discovery picks that fit the vibe.

USER TOP ARTISTS (examples):
- {chr(10).join(top_artists)}

USER TOP TRACKS (examples):
- {chr(10).join(top_tracks)}

Return ONLY a simple numbered list of exactly 30 lines in the format:
1. Track Name — Artist Name
2. ...
"""

def ask_openai_for_playlist(vibe: str, df: pd.DataFrame) -> list[tuple[str, str]]:
    client = get_openai_client()
    top_art = top_items(df["artist"], n=20)
    top_trk = top_items(df["track"], n=20)

    prompt = build_prompt(vibe, top_art, top_trk)

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=1200,
        temperature=0.7,
    )
    text = resp.output_text.strip()

    parsed = []
    for line in text.splitlines():
        # remove bullet/numbering
        line = re.sub(r"^\s*(\d+\.|\-|\•)\s*", "", line).strip()
        if not line:
            continue
        # split on em dash or hyphen
        parts = re.split(r"\s+[—-]\s+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        track, artist = parts[0].strip(), parts[1].strip()
        if track and artist:
            parsed.append((track, artist))
        if len(parsed) == 30:
            break

    return parsed

# ── Clipboard helper (best-effort, no extra deps) ───────────────────────────
def copy_to_clipboard(text: str) -> bool:
    try:
        sysname = platform.system()
        if sysname == "Windows":
            if shutil.which("clip"):
                subprocess.run("clip", input=text, text=True, check=True)
                return True
            # fallback to PowerShell
            ps_cmd = ["powershell", "-NoProfile", "-Command", f"Set-Clipboard -Value @'\n{text}\n'@"]
            subprocess.run(ps_cmd, check=True)
            return True
        elif sysname == "Darwin":
            subprocess.run("pbcopy", input=text, text=True, check=True)
            return True
        else:
            if shutil.which("xclip"):
                subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=True)
                return True
            if shutil.which("xsel"):
                subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=True)
                return True
    except Exception:
        pass
    return False

# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    vibe = input("Describe the playlist vibe (e.g., 'late-night indie chill with some Arctic Monkeys energy'): ").strip()
    if not vibe:
        print("No vibe provided. Exiting.")
        sys.exit(0)

    df = load_history(DATA_PATTERN)
    # standardize names if upstream code used original columns
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "trackName": "track",
        "master_metadata_album_artist_name": "artist",
        "artistName": "artist",
    })

    tracks = ask_openai_for_playlist(vibe, df)
    if not tracks:
        print("Sorry, couldn't parse any tracks from the AI response.")
        sys.exit(1)

    # Build a paste-ready block
    header = f"AI Playlist — {vibe} (30 tracks)"
    lines = [header, "-" * len(header)]
    lines += [f"{i:2}. {t} — {a}" for i, (t, a) in enumerate(tracks, start=1)]
    block = "\n".join(lines)

    # Print to console
    print("\n" + block + "\n")

    # Copy to clipboard (best effort)
    if copy_to_clipboard(block):
        print("✅ Playlist copied to your clipboard. Just paste it wherever you want!")
    else:
        print("ℹ️ Couldn’t copy to clipboard automatically—copy from the console above.")

if __name__ == "__main__":
    main()
