# Music Recommendation System

Predicts how likely a user is to **listen to a song again within one month** of
first hearing it (the KKBox WSDM Music Recommendation Challenge target), and
turns those probabilities into personalized top-N song recommendations.

## How it works

1. **Data** (`src/data.py`) — reads the KKBox layout (`train.csv`, `songs.csv`,
   `members.csv`) from `data/`. If those files are absent it generates a
   synthetic listening log with the same schema and a latent repeat signal, so
   the pipeline runs with zero setup.
2. **Features** (`src/features.py`) — user/song/artist play counts, per-pair
   counts, recency gaps, hour-of-day / day-of-week, account age, and smoothed
   causal target encodings. Every history feature uses only strictly earlier
   events, so there is no leakage from the future.
3. **Model** (`src/model.py`) — LightGBM binary classifier with native
   categorical handling and early stopping; evaluated by AUC / average
   precision / log loss on a **chronological** hold-out split.
4. **Recommendations** (`src/recommend.py`) — candidate songs are appended to
   the log at a future timestamp, scored by the model, and ranked.

## Usage

```bash
pip install -r requirements.txt
python3 main.py --rows 2000000 --rounds 400  # real KKBox data (see below)
python3 main.py --rows 60000 --rounds 300    # quick run / synthetic fallback
python3 main.py --user <msno> --top-n 20     # recommend for a specific user
```

Interactive demo:

```bash
streamlit run app.py
```

Artifacts land in `artifacts/`: `model.txt`, `metrics.json`,
`feature_importance.csv`, `recommendations.csv`.

## Using the real KKBox data

Download the [WSDM – KKBox Music Recommendation
Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge) data
and unzip `train.csv`, `songs.csv`, `members.csv` into `data/`. The loader picks
them up automatically:

```bash
kaggle competitions download -c kkbox-music-recommendation-challenge -p data && unzip 'data/*.zip' -d data
```

KKBox does not ship per-play timestamps; the loader derives a monotonic
pseudo-timestamp from the (chronological) row order so the recency features
still apply. Replace it with the real column if your log has one.

## Results

Real KKBox data, first 2M plays, 400 boosting rounds, chronological 80/20 split:

| metric | value |
| --- | --- |
| AUC | 0.781 |
| average precision | 0.827 |
| log loss | 0.554 |

Top features by gain: `user_repeat_rate`, `artist_name`, `source_type`,
`song_repeat_rate`, `user_play_count`, `source_screen_name`.

The synthetic fallback (60k events) scores AUC 0.633 — enough to confirm the
pipeline works without the download.

`--rows` exists because the full 7.4M-row log needs roughly 16 GB of RAM during
feature construction; 2M rows fits comfortably in 8 GB.
