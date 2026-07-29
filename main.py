"""End-to-end pipeline: load data -> features -> train -> evaluate -> recommend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import load_dataset
from src.features import build_features, time_split
from src.model import evaluate, feature_importance, save, train
from src.recommend import recommend

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Music recommendation system")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--rows", type=int, default=None, help="limit rows / synthetic events")
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--user", default=None, help="user to recommend for (default: most active)")
    args = parser.parse_args()

    ds = load_dataset(args.data_dir, n_rows=args.rows)
    print(
        f"loaded {'synthetic' if ds.synthetic else 'KKBox'} data: "
        f"{len(ds.plays):,} plays, {ds.plays['msno'].nunique():,} users, "
        f"{ds.plays['song_id'].nunique():,} songs, repeat rate {ds.plays['target'].mean():.3f}"
    )

    X, y = build_features(ds)
    X_train, X_valid, y_train, y_valid = time_split(X, y)
    print(f"features: {X.shape[1]} | train {len(X_train):,} | valid {len(X_valid):,}")

    booster = train(X_train, y_train, X_valid, y_valid, num_boost_round=args.rounds)
    metrics = evaluate(booster, X_valid, y_valid)
    print("validation:", json.dumps(metrics, indent=2))
    print("\ntop features by gain:")
    print(feature_importance(booster).head(10).to_string(index=False))

    args.artifacts.mkdir(parents=True, exist_ok=True)
    save(booster, args.artifacts / "model.txt")
    (args.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    feature_importance(booster).to_csv(args.artifacts / "feature_importance.csv", index=False)

    user = args.user or ds.plays["msno"].value_counts().idxmax()
    recs = recommend(booster, ds, user, top_n=args.top_n)
    print(f"\ntop {args.top_n} recommendations for {user}:")
    print(recs.to_string(index=False))
    recs.to_csv(args.artifacts / "recommendations.csv", index=False)


if __name__ == "__main__":
    main()
