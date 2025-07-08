from pathlib import Path
import matplotlib.pyplot as plt

from src.spotify_dna.ingestion import load_streaming_history
from src.spotify_dna.analytics import (
    top_songs,
    top_artists,
    peak_listening_hours,
    recommend_similar_tracks,
    plot_top_artists,
    plot_peak_hours,
)

def humanize_duration(seconds: float) -> str:
    """Convert seconds to 'Xd Yh Zm Ws'."""
    days = int(seconds // 86400)
    rem = seconds % 86400
    hours = int(rem // 3600)
    rem %= 3600
    minutes = int(rem // 60)
    secs = int(rem % 60)
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    if secs or not parts: parts.append(f"{secs}s")
    return " ".join(parts)

def get_unit_choice() -> str:
    choices = {'sec', 'min', 'days', 'mix'}
    prompt = (
        "Select unit for play time:\n"
        "  sec  - seconds\n"
        "  min  - minutes\n"
        "  days - days\n"
        "  mix  - days, hours, minutes, seconds\n"
        "Enter choice (sec/min/days/mix): "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in choices:
            return choice

def main():
    data_dir = Path("data")
    df = load_streaming_history(data_dir)

    # 1) Choose display unit
    unit = get_unit_choice()
    if unit == 'sec':
        factor, label = 1, 'Seconds'
    elif unit == 'min':
        factor, label = 1/60, 'Minutes'
    elif unit == 'days':
        factor, label = 1/86400, 'Days'
    else:
        factor, label = None, 'Duration'

    # 2) Top 10 songs
    songs_df = top_songs(df, n=10)
    print(f"\nTop 10 songs by play time ({label}):")
    if factor is not None:
        songs_df['play_time'] = songs_df['play_seconds'] * factor
        print(
            songs_df[['master_metadata_track_name','play_time']]
            .to_string(index=False, header=['Song', label])
        )
    else:
        songs_df['play_time'] = songs_df['play_seconds'].apply(humanize_duration)
        print(
            songs_df[['master_metadata_track_name','play_time']]
            .to_string(index=False, header=['Song','Play Time'])
        )

    # 3) Top 10 artists
    artists_df = top_artists(df, n=10)
    print(f"\nTop 10 artists by play time ({label}):")
    if factor is not None:
        artists_df['play_time'] = artists_df['play_seconds'] * factor
        print(
            artists_df[['master_metadata_album_artist_name','play_time']]
            .to_string(index=False, header=['Artist', label])
        )
    else:
        artists_df['play_time'] = artists_df['play_seconds'].apply(humanize_duration)
        print(
            artists_df[['master_metadata_album_artist_name','play_time']]
            .to_string(index=False, header=['Artist','Play Time'])
        )

    # 4) Peak listening hours
    peak = peak_listening_hours(df)
    print(f"\nPeak listening hours by play time ({label}):")
    if factor is not None:
        print((peak * factor).to_string())
    else:
        print(peak.apply(humanize_duration).to_string())

    # 5) Seed-based recommendations
    seed = input("\nEnter a seed track (exact name): ").strip()
    try:
        recs = recommend_similar_tracks(df, seed, n=3, window_seconds=300)
    except ValueError as e:
        print(f"\n⚠️  {e}")
    else:
        if not recs:
            print(f"\nNo co-play data for '{seed}'. Try another track.")
        else:
            print(f"\nRecommendations for '{seed}':")
            for track, cnt in recs:
                print(f"  • {track} ({cnt} co-plays)")

    # 6) Show the existing charts
    plot_top_artists(df, n=10)
    plot_peak_hours(df)
    plt.show()

if __name__ == "__main__":
    main()
