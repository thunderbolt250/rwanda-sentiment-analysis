"""
Phase 1 — Fallback: Kaggle Dataset Download
============================================
Use this if Scweet auth token approach doesn't work.
Downloads existing African politics Twitter datasets from Kaggle
to use as a starting point or supplement.

SETUP:
1. Create a free Kaggle account at kaggle.com
2. Go to Account → API → Create New Token
3. Download kaggle.json and place it at ~/.kaggle/kaggle.json
4. Run: pip install kaggle
5. Then run this script
"""

import os
import zipfile
import pandas as pd

OUTPUT_DIR = "../data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Option A: use the Kaggle CLI to download relevant datasets ───
# Run these commands in your terminal one by one:

KAGGLE_COMMANDS = [
    # African news/politics sentiment dataset
    "kaggle datasets download -d adityakharosekar2/twitter-sentiment-analysis-dataset",

    # General multilingual Twitter sentiment (useful for transfer learning)
    "kaggle datasets download -d kazanova/sentiment140",
]

print("Run these commands in your terminal to download datasets:\n")
for cmd in KAGGLE_COMMANDS:
    print(f"  {cmd}")

print("\nThen place the downloaded zip files in ../data/raw/")
print("and run the merge_kaggle_data() function below.\n")


# ─── Option B: merge downloaded Kaggle CSVs ───
def merge_kaggle_data(kaggle_csv_paths: list) -> pd.DataFrame:
    """
    Merge multiple Kaggle CSV datasets into one standardized dataframe.
    Normalizes column names so they match our Scweet output format.
    """
    frames = []

    COLUMN_MAP = {
        # common Kaggle column names → our standard names
        "text"       : "text",
        "tweet"      : "text",
        "content"    : "text",
        "sentiment"  : "label",
        "target"     : "label",
        "polarity"   : "label",
        "date"       : "date",
        "created_at" : "date",
    }

    for path in kaggle_csv_paths:
        print(f"Loading: {path}")
        try:
            df = pd.read_csv(path, encoding="latin-1")
            df.columns = [c.lower().strip() for c in df.columns]
            df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}, inplace=True)

            # keep only columns we care about
            keep = [c for c in ["text", "label", "date", "query"] if c in df.columns]
            df = df[keep].copy()
            df["source"] = os.path.basename(path)
            frames.append(df)
            print(f"  ✓ {len(df)} rows loaded")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not frames:
        print("No data loaded.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset="text", inplace=True)
    output = os.path.join(OUTPUT_DIR, "kaggle_combined.csv")
    combined.to_csv(output, index=False)
    print(f"\n✓ Merged dataset saved to {output} ({len(combined)} rows)")
    return combined


# ─── Option C: manually curated Rwanda keyword filter ───
def filter_rwanda_politics(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a large generic dataset, filter only Rwanda-related political tweets.
    Useful when working with broad Kaggle datasets.
    """
    KEYWORDS = [
        "rwanda", "kagame", "rpf", "kigali",
        "rwandan", "kinyarwanda", "rwandaGov",
        "ubutegetsi", "inzego"
    ]
    pattern = "|".join(KEYWORDS)
    mask = df["text"].str.lower().str.contains(pattern, na=False)
    filtered = df[mask].copy()
    print(f"Filtered {len(filtered)} Rwanda-related tweets from {len(df)} total")
    return filtered


if __name__ == "__main__":
    print("Kaggle fallback script ready.")
    print("Follow the setup instructions above to download datasets.")
