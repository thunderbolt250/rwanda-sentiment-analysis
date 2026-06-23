"""
Phase 1 — Data Collection
Rwanda Political Sentiment Analysis
====================================
Collects tweets about Rwandan politics using Scweet.
No API key needed — uses your X account auth token.

HOW TO GET YOUR AUTH TOKEN:
1. Open Twitter/X in your browser (Chrome recommended)
2. Press F12 to open Developer Tools
3. Go to the "Application" tab
4. On the left, expand "Cookies" → click "https://twitter.com"
5. Find the cookie named "auth_token" and copy its value
6. Paste it below where it says YOUR_AUTH_TOKEN_HERE
"""

import os
import time
import pandas as pd
from datetime import datetime
from Scweet import Scweet

# ─────────────────────────────────────────
#  CONFIGURATION — edit these before running
# ─────────────────────────────────────────

AUTH_TOKEN = "AUTH_TOKEN"   # paste your X auth_token cookie here

SEARCH_QUERIES = [
    "#Rwanda politics",
    "#Kagame",
    "#RPF Rwanda",
    "#RwandaGov",
    "Rwanda government",
    "Rwanda election",
    "Rwanda policy",
    "ubutegetsi Rwanda",       # Kinyarwanda: "governance Rwanda"
    "inzego Rwanda",           # Kinyarwanda: "elections Rwanda"
]

SINCE_DATE  = "2023-01-01"    # start date for tweet collection
UNTIL_DATE  = "2024-12-31"    # end date
LIMIT       = 150             # tweets per query (150 x 9 queries = ~1,350 tweets)
OUTPUT_DIR  = "data/raw"
# ─────────────────────────────────────────


def collect_all_queries():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = []

    print(f"\n{'='*55}")
    print("  Rwanda Political Sentiment — Data Collection")
    print(f"{'='*55}")
    print(f"  Queries   : {len(SEARCH_QUERIES)}")
    print(f"  Limit/qry : {LIMIT}")
    print(f"  Date range: {SINCE_DATE} → {UNTIL_DATE}")
    print(f"{'='*55}\n")

    s = Scweet(auth_token=AUTH_TOKEN, manifest_scrape_on_init=True)

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"[{i}/{len(SEARCH_QUERIES)}] Collecting: '{query}'")
        try:
            tweets = s.search(
                query,
                since=SINCE_DATE,
                until=UNTIL_DATE,
                limit=LIMIT,
                lang="en",        # switch to None to include Kinyarwanda too
                save=False        # we handle saving ourselves
            )

            if tweets and len(tweets) > 0:
                df = pd.DataFrame(tweets)
                df["query"] = query          # track which query sourced each tweet
                all_frames.append(df)
                print(f"    ✓ Collected {len(df)} tweets")
            else:
                print(f"    ⚠ No tweets returned for this query")

            time.sleep(5)   # be polite — avoid hammering the endpoint

        except Exception as e:
            print(f"    ✗ Error on query '{query}': {e}")
            time.sleep(15)  # wait longer after an error
            continue

    if not all_frames:
        print("\n✗ No data collected. Check your auth token and try again.")
        return

    # combine all queries into one dataframe
    combined = pd.concat(all_frames, ignore_index=True)

    # deduplicate by tweet id so overlapping queries don't double-count
    if "id" in combined.columns:
        before = len(combined)
        combined.drop_duplicates(subset="id", inplace=True)
        print(f"\n  Removed {before - len(combined)} duplicate tweets")

    # save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"raw_tweets_{timestamp}.csv")
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"  ✓ Done! {len(combined)} unique tweets saved")
    print(f"  File: {output_path}")
    print(f"{'='*55}\n")

    print_summary(combined)
    return combined


def print_summary(df):
    print("DATASET SUMMARY")
    print(f"  Total tweets     : {len(df)}")
    if "date" in df.columns:
        print(f"  Date range       : {df['date'].min()} → {df['date'].max()}")
    if "query" in df.columns:
        print(f"\n  Tweets per query:")
        for q, count in df["query"].value_counts().items():
            print(f"    {count:>4}  {q}")
    if "language" in df.columns:
        print(f"\n  Languages found  : {df['language'].value_counts().to_dict()}")


if __name__ == "__main__":
    collect_all_queries()
