"""LightGBM repeat-listen classifier."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
}


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    num_boost_round: int = 800,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)
    return lgb.train(
        PARAMS,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )


def evaluate(booster: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = booster.predict(X, num_iteration=booster.best_iteration)
    return {
        "auc": roc_auc_score(y, pred),
        "average_precision": average_precision_score(y, pred),
        "log_loss": log_loss(y, pred),
    }


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
            }
        )
        .sort_values("gain", ascending=False, ignore_index=True)
    )


def save(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path), num_iteration=booster.best_iteration)
