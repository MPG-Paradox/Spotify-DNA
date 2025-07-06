import json
import glob
from pathlib import Path
import pandas as pd

def load_streaming_history(data_dir: Path) -> pd.DataFrame:
    """
    Load all Streaming_History_Audio_*.json files from data_dir,
    concatenate into a DataFrame, parse timestamps,
    and filter only audio entries.
    """
    files = glob.glob(str(data_dir / "Streaming_History_Audio_*.json"))
    dfs = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        df = pd.DataFrame.from_records(records)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    full = pd.concat(dfs, ignore_index=True)
    # parse timestamp as datetime with UTC tz
    full['ts'] = pd.to_datetime(full['ts'], utc=True)
    # keep only rows where a track name exists
    audio_df = full[full['master_metadata_track_name'].notna()].reset_index(drop=True)
    return audio_df
