# 🎵 Spotify-DNA

**Spotify-DNA** is a Python project that analyzes your Spotify listening history, generates insights (top songs/artists, listening patterns), and gives you personalized recommendations.  
It also includes an **AI-powered playlist generator** that creates playlists based on your vibe using the OpenAI API.

---

## 🚀 Features

- Analyze your **Spotify extended streaming history** (from your data export).
- Log in with **Spotify OAuth (PKCE)** to pull data directly from Spotify (no client secret needed).
- View:
  - Top 10 songs by play time
  - Top 10 artists by play time
  - Recommendations (session co-occurrence or Spotify audio features)
  - Optional **listening hours chart** (peak hours visualization, with optional year filter)
- **AI playlist generator** (30 tracks) based on your vibe + your listening history.

---

## 🛠️ Installation
Clone the repo:
``bash
git clone https://github.com/<your-username>/Spotify-DNA.git
cd Spotify-DNA


Create and activate a virtual environment:
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

🔑 Environment Variables
Create a .env file in the project root (this file is .gitignored, so your secrets stay safe):

# Spotify App Client ID (create an app at https://developer.spotify.com/dashboard)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here

# OpenAI API key (https://platform.openai.com)
OPENAI_API_KEY=your_openai_api_key_here


📂 Data
Place your exported Spotify streaming history files in:
./data/Streaming_History_*.json

The project supports both old and new export formats (e.g., msPlayed vs ms_played, endTime vs ts).

🎧 Usage
Run the main explorer:
# from the repo root
py explore.py          # Windows
python3 explore.py     # macOS/Linux

You’ll be prompted to choose an input method:
data → analyze local JSON history in ./data/ (no login)
login → open a browser to log in with Spotify (OAuth PKCE)
Then you’ll be asked:
Select unit for play time: sec / min / days / mix
Whether to generate charts (y/n). If yes, you can optionally filter by a specific year (the latest available year is suggested automatically).
You’ll see:
Top 10 songs by play time
Top 10 artists by play time
Recommendations for a seed track:
In data mode: session co-occurrence (tracks listened in the same sessions)
In login mode: content-based similarity using Spotify audio features
(Optional) A popup chart of listening hours (0–23), optionally filtered to a selected year.



🤖 AI Playlist Generator (30 tracks)
Generate a vibe-based playlist from your history:
# from the repo root
py -m scripts.ai_playlist          # Windows
python -m scripts.ai_playlist      # macOS/Linux

You’ll be asked for a vibe (e.g., late-night indie chill, high-energy gym bangers).
The script uses your listening history + vibe to produce a 30-track list printed to the console.
No Spotify playlist is created automatically (safe for users who don’t know their Spotify IDs).

🖼️ Graphs

Listening activity by hour bar chart:
X-axis: 0–23 (clock hours)
Y-axis: total hours listened
Clean labels, value annotations, gridlines, and configurable year filter
The chart is shown as a popup window (Matplotlib interactive).

🔒 Security & Git Hygiene
Never hard-code secrets in code. Use .env.
.gitignore includes:
.cache*
*.env
.env
.env.local
.DS_Store
outputs/

Spotify login is via official OAuth PKCE — no password handling.

🧪 Quick Checks

Verify OpenAI setup:
py -m scripts.test_openai          # Windows
python -m scripts.test_openai      # macOS/Linux

Run explorer with local data:
py explore.py
# choose: data

Run explorer with Spotify login:
py explore.py
# choose: login
# browser opens → log in → consent → window closes → script continues


📜 License
MIT — see LICENSE.
::contentReference[oaicite:0]{index=0}
