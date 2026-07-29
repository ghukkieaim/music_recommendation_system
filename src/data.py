"""Dataset loading for the music recommendation system.

Supports the KKBox WSDM Music Recommendation Challenge layout
(``train.csv``, ``songs.csv``, ``members.csv``) and falls back to a
synthetic generator so the pipeline is runnable without the Kaggle download.

The prediction target follows the challenge definition: ``1`` when the user
listened to the song again within one month of the first observed play.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KKBOX_FILES = ("train.csv", "songs.csv", "members.csv")


@dataclass
class Dataset:
    plays: pd.DataFrame  # msno, song_id, timestamp, source_*, target
    songs: pd.DataFrame  # song_id, song_length, genre_ids, artist_name, language
    members: pd.DataFrame  # msno, city, bd, gender, registered_via, registration_init_time
    synthetic: bool


def load_dataset(data_dir: Path, n_rows: int | None = None, seed: int = 0) -> Dataset:
    if all((data_dir / name).exists() for name in KKBOX_FILES):
        return _load_kkbox(data_dir, n_rows)
    return generate_synthetic(seed=seed, n_events=n_rows or 120_000)


def _load_kkbox(data_dir: Path, n_rows: int | None) -> Dataset:
    plays = pd.read_csv(data_dir / "train.csv", nrows=n_rows)
    songs = pd.read_csv(data_dir / "songs.csv")
    members = pd.read_csv(data_dir / "members.csv")

    if "timestamp" not in plays.columns:
        # KKBox ships plays in chronological order without explicit stamps;
        # use the row order as a monotonic pseudo-timestamp spread over the
        # ~6 month collection window.
        span = pd.Timedelta(days=180)
        step = span / max(len(plays), 1)
        plays["timestamp"] = pd.Timestamp("2017-01-01") + step * np.arange(len(plays))
    else:
        plays["timestamp"] = pd.to_datetime(plays["timestamp"])

    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"], format="%Y%m%d", errors="coerce"
    )
    return Dataset(plays=plays, songs=songs, members=members, synthetic=False)


def generate_synthetic(
    seed: int = 0,
    n_users: int = 2_000,
    n_songs: int = 5_000,
    n_events: int = 120_000,
) -> Dataset:
    """Generate a KKBox-shaped listening log with a learnable repeat signal."""
    rng = np.random.default_rng(seed)

    genres = [f"{rng.integers(100, 2000)}" for _ in range(40)]
    artists = [f"artist_{i}" for i in range(300)]
    songs = pd.DataFrame(
        {
            "song_id": [f"s{i}" for i in range(n_songs)],
            "song_length": rng.normal(230_000, 60_000, n_songs).clip(30_000).astype(int),
            "genre_ids": rng.choice(genres, n_songs),
            "artist_name": rng.choice(artists, n_songs),
            "language": rng.choice([3, 10, 17, 24, 31, 52], n_songs),
        }
    )
    song_quality = rng.beta(2, 5, n_songs)  # latent "stickiness" per song

    members = pd.DataFrame(
        {
            "msno": [f"u{i}" for i in range(n_users)],
            "city": rng.integers(1, 22, n_users),
            "bd": rng.integers(0, 60, n_users),
            "gender": rng.choice(["male", "female", None], n_users),
            "registered_via": rng.choice([3, 4, 7, 9], n_users),
            "registration_init_time": pd.Timestamp("2012-01-01")
            + pd.to_timedelta(rng.integers(0, 1800, n_users), unit="D"),
        }
    )
    user_loyalty = rng.beta(2, 4, n_users)  # latent replay propensity per user

    user_idx = rng.integers(0, n_users, n_events)
    # Zipf-ish popularity so a few songs dominate, as in real logs.
    pop = rng.zipf(1.4, n_events) % n_songs
    tabs = ["my library", "discover", "search", "radio", "listen with"]
    screens = ["Local playlist more", "Online playlist more", "Radio", "Album more", "Search"]
    types = ["local-library", "online-playlist", "radio", "album", "song-based-playlist"]

    plays = pd.DataFrame(
        {
            "msno": members["msno"].to_numpy()[user_idx],
            "song_id": songs["song_id"].to_numpy()[pop],
            "source_system_tab": rng.choice(tabs, n_events),
            "source_screen_name": rng.choice(screens, n_events),
            "source_type": rng.choice(types, n_events),
            "timestamp": pd.Timestamp("2017-01-01")
            + pd.to_timedelta(rng.integers(0, 180 * 24 * 60, n_events), unit="m"),
        }
    )

    from_library = (plays["source_system_tab"] == "my library").to_numpy()
    logit = (
        -1.1
        + 3.0 * user_loyalty[user_idx]
        + 2.6 * song_quality[pop]
        + 0.8 * from_library
        + rng.normal(0, 0.5, n_events)
    )
    prob = 1 / (1 + np.exp(-logit))
    plays["target"] = (rng.random(n_events) < prob).astype(int)

    plays = plays.sort_values("timestamp", ignore_index=True)
    return Dataset(plays=plays, songs=songs, members=members, synthetic=True)
