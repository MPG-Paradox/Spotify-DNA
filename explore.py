from pathlib import Path
import matplotlib.pyplot as plt

from src.spotify_dna.ingestion import load_streaming_history
from src.spotify_dna.analytics import (
    top_songs,
    top_artists,
    peak_listening_hours,
    top_song_pairs,
    recommend_similar_tracks,
    plot_top_artists,
    plot_peak_hours,
)

def main():
    data_dir = Path("data")
    df = load_streaming_history(data_dir)

    # 1) Top 10 songs
    print("\nTop 10 songs by play time (seconds):")
    print(top_songs(df, n=10).to_string(index=False))

    # 2) Top 10 artists
    print("\nTop 10 artists by play time (seconds):")
    print(top_artists(df, n=10).to_string(index=False))

    # 3) Peak listening hours
    print("\nPeak listening hours (seconds):")
    print(peak_listening_hours(df).to_string())

    # 4) Top 5 song-pairs
    print("\nTop 5 song-pairs by count (within 5 min):")
    print(top_song_pairs(df, n=5, window_seconds=300).to_string(index=False))

    # 5) Seed-based recommendations
    seed = input("\nEnter a seed track (exact name as in your history): ").strip()
    try:
        recs = recommend_similar_tracks(df, seed, n=3, window_seconds=300)
    except ValueError as e:
        print(f"\n⚠️  {e}")
    else:
        if not recs:
            print(f"\nNo co-play data found for '{seed}'. Try a different track.")
        else:
            print(f"\nTracks to go with '{seed}':")
            for track, cnt in recs:
                print(f"  • {track} ({cnt} co-plays)")

    # 6) Charts (in raw seconds)
    plot_top_artists(df, n=10)
    plot_peak_hours(df)
    plt.show()

if __name__ == "__main__":
    main()
