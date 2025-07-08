from pathlib import Path
import matplotlib.pyplot as plt

from src.spotify_dna.ingestion import load_streaming_history
from src.spotify_dna.analytics import (
    top_artists,
    plot_top_artists,
    peak_listening_hours,
    plot_peak_hours,
)

def humanize_duration(seconds: float) -> str:
    """
    Turn a raw number of seconds into "Xd Yh Zm Ws" form.
    """
    days = int(seconds // 86400)
    rem = seconds % 86400
    hours = int(rem // 3600)
    rem %= 3600
    minutes = int(rem // 60)
    secs = int(rem % 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    # Always show seconds if nothing else or if >0
    if secs or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)

def get_unit_choice() -> str:
    """
    Prompt until the user chooses one of: sec, min, days, mix
    """
    choices = {'sec', 'min', 'days', 'mix'}
    prompt = (
        "Select unit for play time:\n"
        "  sec  - seconds\n"
        "  min  - minutes\n"
        "  days - days\n"
        "  mix  - days, hours, minutes, seconds\n"
        "Enter choice (sec/min/days/mix): "
    )
    choice = ''
    while choice not in choices:
        choice = input(prompt).strip().lower()
    return choice

def main():
    data_dir = Path("data")
    df = load_streaming_history(data_dir)

    # 1) Ask for unit
    unit = get_unit_choice()
    if unit == 'sec':
        factor = 1
        label = 'Seconds'
    elif unit == 'min':
        factor = 1/60
        label = 'Minutes'
    elif unit == 'days':
        factor = 1/86400
        label = 'Days'
    else:  # mix
        factor = None
        label = 'Duration'

    # 2) Top Artists table
    artists_df = top_artists(df, n=10)
    print(f"\nTop 10 artists by play time ({label}):")
    if factor is not None:
        # numeric conversion
        artists_df['play_time'] = artists_df['play_seconds'] * factor
        print(
            artists_df
            .loc[:, ['master_metadata_album_artist_name', 'play_time']]
            .to_string(index=False, header=['Artist', label])
        )
    else:
        # humanized mix
        artists_df['play_time'] = artists_df['play_seconds'].apply(humanize_duration)
        print(
            artists_df
            .loc[:, ['master_metadata_album_artist_name', 'play_time']]
            .to_string(index=False, header=['Artist', 'Play Time'])
        )

    # 3) Peak Listening Hours table
    peak = peak_listening_hours(df)
    print(f"\nPeak listening hours by play time ({label}):")
    if factor is not None:
        peak_conv = peak * factor
        print(peak_conv.to_string())
    else:
        peak_conv = peak.apply(humanize_duration)
        print(peak_conv.to_string())

    # 4) Show your charts (still in raw seconds for now)
    fig1 = plot_top_artists(df, n=10)
    fig2 = plot_peak_hours(df)
    plt.show()

if __name__ == "__main__":
    main()
