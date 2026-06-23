"""
Phase 2 — Preprocessing
Rwanda Political Sentiment Analysis
=====================================
Cleans and prepares raw tweets for labeling and model training.

Run from the project root:
    python src/preprocess.py
"""

import re
import pandas as pd
from langdetect import detect, LangDetectException

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
INPUT_FILE  = "data/raw/combined_tweets.csv"
OUTPUT_FILE = "data/processed/cleaned_tweets.csv"
MIN_WORDS   = 4      # drop tweets shorter than this after cleaning
# ─────────────────────────────────────────


def clean_text(text: str) -> str:
    """Remove noise from a raw tweet."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#\w+", "", text)                     # hashtags
    text = re.sub(r"RT\s+", "", text)                    # retweet prefix
    text = re.sub(r"[^\w\s',!?.-]", " ", text)          # special chars (keep punctuation)
    text = re.sub(r"\s+", " ", text).strip()             # extra whitespace
    return text


def detect_language(text: str) -> str:
    """Detect language of a tweet. Returns 'en', 'rw', or 'unknown'."""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"


def preprocess(input_file: str, output_file: str) -> pd.DataFrame:
    import os
    os.makedirs("data/processed", exist_ok=True)

    print(f"\n{'='*55}")
    print("  Phase 2 — Preprocessing")
    print(f"{'='*55}")

    # ── load ──────────────────────────────────────────────────
    df = pd.read_csv(input_file)
    print(f"\n  Loaded        : {len(df)} tweets")

    # ── drop rows with no text ────────────────────────────────
    df.dropna(subset=["text"], inplace=True)
    print(f"  After dropna  : {len(df)} tweets")

    # ── clean text ────────────────────────────────────────────
    df["clean_text"] = df["text"].apply(clean_text)

    # ── drop tweets that are too short after cleaning ─────────
    df = df[df["clean_text"].apply(lambda x: len(x.split()) >= MIN_WORDS)]
    print(f"  After length filter ({MIN_WORDS}+ words): {len(df)} tweets")

    # ── drop duplicate clean texts ────────────────────────────
    before = len(df)
    df.drop_duplicates(subset="clean_text", inplace=True)
    print(f"  After dedup   : {len(df)} tweets (removed {before - len(df)})")

    # ── detect language ───────────────────────────────────────
    print("\n  Detecting languages (this may take a minute)...")
    df["language"] = df["clean_text"].apply(detect_language)
    lang_counts = df["language"].value_counts()
    print(f"  Language breakdown:")
    for lang, count in lang_counts.items():
        print(f"    {lang:<10} {count}")

    # ── parse timestamp ───────────────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"]  = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    # ── add placeholder label column (for Phase 3) ────────────
    df["label"] = None   # will be filled in Phase 3

    # ── select and reorder final columns ─────────────────────
    keep = ["tweet_id", "clean_text", "text", "language",
            "timestamp", "date", "month",
            "likes", "retweets", "comments",
            "query", "label"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].reset_index(drop=True)

    # ── save ──────────────────────────────────────────────────
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"  ✓ Done! {len(df)} clean tweets saved")
    print(f"  File: {output_file}")
    print(f"{'='*55}\n")

    print_sample(df)
    return df


def print_sample(df: pd.DataFrame):
    print("SAMPLE CLEANED TWEETS")
    print("-" * 55)
    for _, row in df.head(5).iterrows():
        print(f"  [{row.get('language', '?')}] {row['clean_text'][:100]}")
    print("-" * 55)


if __name__ == "__main__":
    preprocess(INPUT_FILE, OUTPUT_FILE)
