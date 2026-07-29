"""Feature engineering from listening history and timestamps.

All history-derived features are strictly causal: for each play event only
events that happened *earlier* contribute, so the training set contains no
information from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset

CATEGORICAL = [
    "source_system_tab",
    "source_screen_name",
    "source_type",
    "genre_ids",
    "artist_name",
    "language",
    "city",
    "registered_via",
    "gender",
]


def _expanding_prior_mean(values: pd.Series, group: pd.Series, prior_weight: float = 20.0):
    """Smoothed mean of `values` over the group's strictly previous rows."""
    grouped = values.groupby(group)
    cum_sum = grouped.cumsum() - values
    cum_cnt = grouped.cumcount()
    global_mean = values.mean()
    return (cum_sum + prior_weight * global_mean) / (cum_cnt + prior_weight)


def build_features(ds: Dataset) -> tuple[pd.DataFrame, pd.Series]:
    df = ds.plays.sort_values("timestamp", ignore_index=True).copy()
    df = df.merge(ds.songs, on="song_id", how="left").merge(ds.members, on="msno", how="left")

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["days_since_registration"] = (ts - df["registration_init_time"]).dt.days

    # Past interaction counts.
    df["user_play_count"] = df.groupby("msno").cumcount()
    df["song_play_count"] = df.groupby("song_id").cumcount()
    df["artist_play_count"] = df.groupby("artist_name").cumcount()
    df["user_song_play_count"] = df.groupby(["msno", "song_id"]).cumcount()
    df["user_artist_play_count"] = df.groupby(["msno", "artist_name"]).cumcount()

    # Recency signals (hours since the user's / this pair's previous play).
    df["hours_since_user_prev"] = (
        ts - df.groupby("msno")["timestamp"].shift()
    ).dt.total_seconds() / 3600
    df["hours_since_song_prev"] = (
        ts - df.groupby(["msno", "song_id"])["timestamp"].shift()
    ).dt.total_seconds() / 3600

    # Causal target encodings.
    y = df["target"].astype(float)
    df["user_repeat_rate"] = _expanding_prior_mean(y, df["msno"])
    df["song_repeat_rate"] = _expanding_prior_mean(y, df["song_id"])
    df["artist_repeat_rate"] = _expanding_prior_mean(y, df["artist_name"].fillna("unknown"))

    df["song_length_min"] = df["song_length"] / 60_000
    df["bd"] = df["bd"].where(df["bd"].between(5, 90))

    for col in CATEGORICAL:
        if col in df:
            df[col] = df[col].astype("category")

    numeric = [
        "hour",
        "dayofweek",
        "is_weekend",
        "days_since_registration",
        "user_play_count",
        "song_play_count",
        "artist_play_count",
        "user_song_play_count",
        "user_artist_play_count",
        "hours_since_user_prev",
        "hours_since_song_prev",
        "user_repeat_rate",
        "song_repeat_rate",
        "artist_repeat_rate",
        "song_length_min",
        "bd",
    ]
    feature_cols = [c for c in CATEGORICAL + numeric if c in df]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return X, df["target"].astype(int)


def time_split(X: pd.DataFrame, y: pd.Series, valid_frac: float = 0.2):
    """Chronological split (rows are already time-ordered)."""
    cut = int(len(X) * (1 - valid_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
