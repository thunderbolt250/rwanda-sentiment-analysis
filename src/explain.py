"""
Phase 5 — Explainability
Rwanda Political Sentiment Analysis
=====================================
Uses SHAP to explain model predictions:
- Which words pushed a tweet toward positive/negative/neutral?
- Global feature importance across the whole dataset
- Per-tweet explanations

Run from the project root:
    python src/explain.py
"""

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
INPUT_FILE   = "data/processed/final_labeled_tweets.csv"
MODEL_FILE   = "models/logistic_regression.pkl"
TFIDF_FILE   = "models/tfidf_vectorizer.pkl"
OUTPUT_DIR   = "results/explainability"
SAMPLE_N     = 100    # number of tweets to explain (keep small for speed)
# ─────────────────────────────────────────


def load_artifacts():
    """Load model, vectorizer, and data."""
    print("  Loading model and data...")
    lr    = joblib.load(MODEL_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    df    = pd.read_csv(INPUT_FILE)
    df    = df.dropna(subset=["final_label", "clean_text"])
    df["final_label"] = df["final_label"].str.lower().str.strip()
    df    = df[df["final_label"].isin(["positive", "neutral", "negative"])]
    print(f"  Loaded {len(df)} tweets")
    return lr, tfidf, df


def run_shap(lr, tfidf, df):
    """Run SHAP analysis on logistic regression + TF-IDF."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # sample tweets for explanation
    sample_df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)
    X_sample  = tfidf.transform(sample_df["clean_text"].astype(str))

    print("  Running SHAP explainer (this takes ~1 minute)...")
    explainer   = shap.LinearExplainer(lr, X_sample,
                                       feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)

    feature_names = tfidf.get_feature_names_out()
    class_names   = lr.classes_

    print(f"  Classes: {list(class_names)}")
    print(f"  SHAP values shape: {np.array(shap_values).shape}")

    return shap_values, X_sample, feature_names, class_names, sample_df


def plot_global_importance(shap_values, feature_names, class_names):
    """Bar chart of top words driving each sentiment class."""
    print("\n  Plotting global feature importance...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Top Words Driving Each Sentiment Class\n(Rwanda Political Tweets)",
                 fontsize=14, fontweight="bold", y=1.02)

    colors = {"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#2ECC71"}

    for i, (cls, ax) in enumerate(zip(class_names, axes)):
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else:
            sv = shap_values[:, :, i] if shap_values.ndim == 3 else shap_values

        # mean absolute SHAP value per feature
        mean_shap = np.abs(sv).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[-15:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        top_vals  = mean_shap[top_idx]

        color = colors.get(cls, "#3498DB")
        bars  = ax.barh(range(len(top_words)), top_vals[::-1],
                        color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words[::-1], fontsize=10)
        ax.set_title(f"{cls.upper()}", fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # value labels on bars
        for bar, val in zip(bars, top_vals[::-1]):
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/global_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")


def plot_sample_explanations(shap_values, X_sample,
                              feature_names, class_names, sample_df):
    """Show word-level explanation for one tweet per class."""
    print("\n  Plotting per-tweet explanations...")

    for cls_idx, cls_name in enumerate(class_names):
        # find a tweet actually predicted as this class
        if isinstance(shap_values, list):
            sv = shap_values[cls_idx]
        else:
            sv = shap_values[:, :, cls_idx] if shap_values.ndim == 3 else shap_values

        # pick tweet with highest SHAP sum for this class
        class_scores = sv.sum(axis=1)
        tweet_idx    = np.argmax(class_scores)
        tweet_text   = sample_df.iloc[tweet_idx]["clean_text"]
        tweet_shap   = sv[tweet_idx]

        # top positive and negative contributors
        top_pos_idx = np.argsort(tweet_shap)[-8:][::-1]
        top_neg_idx = np.argsort(tweet_shap)[:8]

        words  = ([feature_names[j] for j in top_pos_idx] +
                  [feature_names[j] for j in top_neg_idx])
        values = ([tweet_shap[j] for j in top_pos_idx] +
                  [tweet_shap[j] for j in top_neg_idx])

        # remove zeros
        pairs  = [(w, v) for w, v in zip(words, values) if abs(v) > 0.001]
        if not pairs:
            continue
        words, values = zip(*pairs)

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in values]
        ax.barh(range(len(words)), values, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (positive = pushes toward this class)", fontsize=10)
        ax.set_title(
            f"Word Contributions → '{cls_name.upper()}' prediction\n"
            f"Tweet: \"{tweet_text[:80]}...\"",
            fontsize=11, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        out = f"{OUTPUT_DIR}/tweet_explanation_{cls_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {out}")


def print_top_words(shap_values, feature_names, class_names):
    """Print top 10 words per class to terminal."""
    print(f"\n{'='*55}")
    print("  TOP WORDS PER SENTIMENT CLASS")
    print(f"{'='*55}")

    for i, cls in enumerate(class_names):
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else:
            sv = shap_values[:, :, i] if shap_values.ndim == 3 else shap_values

        mean_shap = np.abs(sv).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[-10:][::-1]

        print(f"\n  {cls.upper()}:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:>2}. {feature_names[idx]:<20} {mean_shap[idx]:.4f}")


def save_shap_summary(shap_values, feature_names, class_names):
    """Save SHAP summary as CSV for use in dashboard."""
    rows = []
    for i, cls in enumerate(class_names):
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else:
            sv = shap_values[:, :, i] if shap_values.ndim == 3 else shap_values

        mean_shap = np.abs(sv).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[-20:][::-1]
        for idx in top_idx:
            rows.append({
                "class"      : cls,
                "word"       : feature_names[idx],
                "importance" : round(float(mean_shap[idx]), 5)
            })

    summary_df = pd.DataFrame(rows)
    out = f"{OUTPUT_DIR}/shap_word_importance.csv"
    summary_df.to_csv(out, index=False)
    print(f"\n  ✓ SHAP summary saved: {out}")
    return summary_df


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  Phase 5 — Explainability (SHAP)")
    print(f"{'='*55}\n")

    # load
    lr, tfidf, df = load_artifacts()

    # run SHAP
    shap_values, X_sample, feature_names, class_names, sample_df = run_shap(
        lr, tfidf, df
    )

    # terminal output
    print_top_words(shap_values, feature_names, class_names)

    # plots
    plot_global_importance(shap_values, feature_names, class_names)
    plot_sample_explanations(shap_values, X_sample,
                              feature_names, class_names, sample_df)

    # save CSV for dashboard
    save_shap_summary(shap_values, feature_names, class_names)

    print(f"\n{'='*55}")
    print("  ✓ Phase 5 Complete!")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*55}\n")
