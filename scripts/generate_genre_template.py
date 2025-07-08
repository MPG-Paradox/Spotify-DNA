from pathlib import Path
import pandas as pd
from src.spotify_dna.ingestion import load_streaming_history

def main():
    # Load your streaming history
    df = load_streaming_history(Path("data"))
    # Grab every unique track URI
    uris = df['spotify_track_uri'].dropna().unique()
    # Build a DataFrame with an empty 'genre' column
    template = pd.DataFrame({
        'spotify_track_uri': uris,
        'genre': ''
    })
    # Save it for you to fill in
    template.to_csv(Path("data/track_genres_template.csv"), index=False)
    print(f"Generated template with {len(uris)} URIs → data/track_genres_template.csv")

if __name__ == "__main__":
    main()
