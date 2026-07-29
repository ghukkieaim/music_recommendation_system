"""Streamlit demo: train the repeat-listen model and browse recommendations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_dataset
from src.features import build_features, time_split
from src.model import evaluate, feature_importance, train
from src.recommend import recommend

ROOT = Path(__file__).parent

st.set_page_config(page_title="Music Recommendation System", layout="wide")
st.title("Music Recommendation System")
st.caption("Predicts repeat listens within a month and ranks songs per user.")


@st.cache_resource(show_spinner="Loading data and training model...")
def load_and_train(rows: int, rounds: int):
    ds = load_dataset(ROOT / "data", n_rows=rows)
    X, y = build_features(ds)
    X_train, X_valid, y_train, y_valid = time_split(X, y)
    booster = train(X_train, y_train, X_valid, y_valid, num_boost_round=rounds)
    return ds, booster, evaluate(booster, X_valid, y_valid)


with st.sidebar:
    st.header("Training")
    rows = st.slider("Events", 20_000, 200_000, 60_000, step=20_000)
    rounds = st.slider("Boosting rounds", 100, 1_000, 300, step=100)
    top_n = st.slider("Recommendations", 5, 30, 10)

ds, booster, metrics = load_and_train(rows, rounds)

source = "synthetic" if ds.synthetic else "KKBox"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Plays", f"{len(ds.plays):,}", help=f"{source} data")
c2.metric("Validation AUC", f"{metrics['auc']:.3f}")
c3.metric("Average precision", f"{metrics['average_precision']:.3f}")
c4.metric("Repeat rate", f"{ds.plays['target'].mean():.3f}")

users = ds.plays["msno"].value_counts()
user = st.selectbox("User", users.index[:200], format_func=lambda u: f"{u} ({users[u]} plays)")

recs = recommend(booster, ds, user, top_n=top_n)
st.subheader(f"Top {top_n} recommendations")
st.dataframe(
    recs.style.format({"score": "{:.3f}"}).background_gradient(subset=["score"], cmap="Greens"),
    use_container_width=True,
)

with st.expander("Listening history"):
    history = ds.plays[ds.plays["msno"] == user].sort_values("timestamp", ascending=False)
    st.dataframe(history.head(50), use_container_width=True)

with st.expander("Feature importance"):
    imp = feature_importance(booster).head(20)
    st.bar_chart(imp.set_index("feature")["gain"])
    st.dataframe(pd.DataFrame(metrics, index=["value"]).T, use_container_width=True)
