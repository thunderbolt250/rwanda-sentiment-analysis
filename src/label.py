"""
Phase 3 — Labeling
Rwanda Political Sentiment Analysis
=====================================
Auto-labels tweets using VADER sentiment analyzer.
Produces a CSV ready for manual review and model training.

Run from the project root:
    python src/label.py
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
INPUT_FILE        = "data/processed/cleaned_tweets.csv"
OUTPUT_FILE       = "data/processed/labeled_tweets.csv"
REVIEW_FILE       = "data/processed/manual_review_sample.csv"
REVIEW_SAMPLE_N   = 200     # tweets to flag for manual review
POSITIVE_THRESH   =  0.05   # compound score above this → positive
NEGATIVE_THRESH   = -0.05   # compound score below this → negative
                            # between the two → neutral
# ─────────────────────────────────────────


def vader_label(compound: float) -> str:
    """Convert VADER compound score to sentiment label."""
    if compound >= POSITIVE_THRESH:
        return "positive"
    elif compound <= NEGATIVE_THRESH:
        return "negative"
    else:
        return "neutral"


def label_tweets(input_file: str, output_file: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  Phase 3 — Labeling")
    print(f"{'='*55}\n")

    # ── load ──────────────────────────────────────────────────
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} cleaned tweets")

    # ── run VADER ─────────────────────────────────────────────
    analyzer = SentimentIntensityAnalyzer()

    print("  Running VADER sentiment scoring...")
    scores = df["clean_text"].apply(lambda t: analyzer.polarity_scores(str(t)))

    df["vader_neg"]      = scores.apply(lambda s: s["neg"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_compound"] = scores.apply(lambda s: s["compound"])

    # ── assign auto label ─────────────────────────────────────
    df["label"] = df["vader_compound"].apply(vader_label)

    # ── add manual_label column (blank — for your corrections) 
    df["manual_label"] = None

    # ── print distribution ────────────────────────────────────
    print(f"\n  AUTO-LABEL DISTRIBUTION")
    print(f"  {'-'*35}")
    counts = df["label"].value_counts()
    total  = len(df)
    for label, count in counts.items():
        bar = "█" * (count // 10)
        print(f"  {label:<10} {count:>4}  ({count/total*100:.1f}%)  {bar}")

    # ── flag low-confidence tweets for manual review ──────────
    # low confidence = compound score close to 0 (near the thresholds)
    df["confidence"] = df["vader_compound"].abs()
    low_conf = df.nsmallest(REVIEW_SAMPLE_N, "confidence")

    review_df = low_conf[["tweet_id", "clean_text", "label",
                           "vader_compound", "manual_label"]].copy()
    review_df.to_csv(REVIEW_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  ✎  {len(review_df)} low-confidence tweets saved for manual review")
    print(f"     File: {REVIEW_FILE}")
    print(f"     → Open this CSV, read each tweet, and fill in 'manual_label'")
    print(f"       with: positive / neutral / negative")

    # ── save full labeled dataset ──────────────────────────────
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"  ✓ Done! {len(df)} labeled tweets saved")
    print(f"  File: {output_file}")
    print(f"{'='*55}\n")

    print_samples(df)
    return df


def print_samples(df: pd.DataFrame):
    """Print 2 examples from each sentiment class."""
    print("SAMPLE LABELS")
    print("-" * 55)
    for label in ["positive", "neutral", "negative"]:
        subset = df[df["label"] == label].head(2)
        for _, row in subset.iterrows():
            score = row["vader_compound"]
            text  = row["clean_text"][:90]
            print(f"  [{label:<8} | {score:+.3f}]  {text}")
        print()


if __name__ == "__main__":
    label_tweets(INPUT_FILE, OUTPUT_FILE)
