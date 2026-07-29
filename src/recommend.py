"""Top-N song recommendations for a user.

Candidate events are appended to the listening log at a future timestamp and
scored with the trained model; because the feature builder is causal, the
candidate rows see the user's full history and nothing else.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from .data import Dataset
from .features import build_features


def recommend(
    booster: lgb.Booster,
    ds: Dataset,
    user: str,
    top_n: int = 10,
    n_candidates: int = 500,
    source_system_tab: str = "my library",
    source_screen_name: str = "Local playlist more",
    source_type: str = "local-library",
) -> pd.DataFrame:
    plays = ds.plays
    if user not in set(plays["msno"]):
        raise ValueError(f"unknown user: {user}")

    popular = plays["song_id"].value_counts().head(n_candidates).index
    heard = plays.loc[plays["msno"] == user, "song_id"]
    candidates = pd.Index(sorted(set(popular) | set(heard)))

    at = plays["timestamp"].max() + pd.Timedelta(hours=1)
    candidate_rows = pd.DataFrame(
        {
            "msno": user,
            "song_id": candidates,
            "source_system_tab": source_system_tab,
            "source_screen_name": source_screen_name,
            "source_type": source_type,
            "timestamp": at,
            "target": 0,  # placeholder, never used for scoring
        }
    )

    scoring_ds = Dataset(
        plays=pd.concat([plays, candidate_rows], ignore_index=True),
        songs=ds.songs,
        members=ds.members,
        synthetic=ds.synthetic,
    )
    X, _ = build_features(scoring_ds)
    X_cand = X.iloc[-len(candidate_rows) :]
    scores = booster.predict(X_cand, num_iteration=booster.best_iteration)

    out = candidate_rows[["song_id"]].copy()
    out["score"] = scores
    out = out.merge(ds.songs[["song_id", "artist_name", "genre_ids"]], on="song_id", how="left")
    out["already_heard"] = out["song_id"].isin(set(heard))
    return out.sort_values("score", ascending=False, ignore_index=True).head(top_n)
