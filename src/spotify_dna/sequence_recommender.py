from typing import List
from datetime import timedelta
import pandas as pd
from gensim.models import Word2Vec

def _build_sessions(df: pd.DataFrame, max_gap_minutes: int = 30) -> List[List[str]]:
    """
    Breaks sorted play history into sessions: a new session whenever gap > max_gap_minutes.
    Returns a list of sessions, each a list of track names.
    """
    df2 = df.sort_values('ts').reset_index(drop=True)
    sessions = []
    current = []
    prev_ts = None

    for row in df2.itertuples():
        if prev_ts is not None and (row.ts - prev_ts) > timedelta(minutes=max_gap_minutes):
            if current:
                sessions.append(current)
            current = []
        current.append(row.master_metadata_track_name)
        prev_ts = row.ts

    if current:
        sessions.append(current)
    return sessions

class SeqRecommender:
    """
    Trains a Word2Vec model over your listening sessions,
    then finds nearest neighbors in the embedding space.
    """
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model: Word2Vec = None

    def fit(self, df: pd.DataFrame, max_gap_minutes: int = 30):
        sessions = _build_sessions(df, max_gap_minutes)
        # Train Word2Vec on sessions (each session is a 'sentence' of track names)
        self.model = Word2Vec(
            sentences=sessions,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            seed=42,
        )

    def similar(self, seed: str, n: int = 3) -> List[tuple[str, float]]:
        """
        Returns up to n most-similar track names to the seed.
        Raises if seed not in vocabulary.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if seed not in self.model.wv:
            raise ValueError(f"'{seed}' not found in session vocabulary.")
        return self.model.wv.most_similar(seed, topn=n)
