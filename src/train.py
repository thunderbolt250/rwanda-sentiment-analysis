"""
Phase 4 — Model Training
Rwanda Political Sentiment Analysis
=====================================
Trains two models:
  1. Baseline  — Logistic Regression + TF-IDF
  2. Advanced  — AfroXLMR (multilingual transformer)

Run from the project root:
    python src/train.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
INPUT_FILE   = "data/processed/final_labeled_tweets.csv"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"
RANDOM_SEED  = 42
TEST_SIZE    = 0.2       # 80% train, 20% test
LABEL_COL    = "final_label"
TEXT_COL_LR  = "clean_text"   # for logistic regression
TEXT_COL_TR  = "text"         # raw text for transformer
# ─────────────────────────────────────────


def load_data(input_file: str):
    """Load and validate the labeled dataset."""
    df = pd.read_csv(input_file)

    # drop rows with missing labels or text
    df = df.dropna(subset=[LABEL_COL, TEXT_COL_LR])

    # normalize labels to lowercase
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()

    # keep only valid labels
    valid = ["positive", "neutral", "negative"]
    df = df[df[LABEL_COL].isin(valid)]

    print(f"  Loaded {len(df)} labeled tweets")
    print(f"  Label distribution:")
    counts = df[LABEL_COL].value_counts()
    for label, count in counts.items():
        print(f"    {label:<10} {count}")

    return df


# ═══════════════════════════════════════════════════════
#  MODEL 1 — LOGISTIC REGRESSION + TF-IDF (BASELINE)
# ═══════════════════════════════════════════════════════

def train_logistic_regression(df: pd.DataFrame):
    print(f"\n{'─'*55}")
    print("  MODEL 1 — Logistic Regression + TF-IDF")
    print(f"{'─'*55}")

    X = df[TEXT_COL_LR].astype(str)
    y = df[LABEL_COL]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF vectorization
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    custom_stops = list(ENGLISH_STOP_WORDS) + ["amp", "rt", "via", "http", "https"]

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words=custom_stops
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # train logistic regression
    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced"   # handles class imbalance
    )
    lr.fit(X_train_vec, y_train)

    # evaluate
    y_pred = lr.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["negative", "neutral", "positive"],
                                zero_division=0))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred,
                          labels=["negative", "neutral", "positive"])
    cm_df = pd.DataFrame(cm,
                         index=["actual_neg", "actual_neu", "actual_pos"],
                         columns=["pred_neg", "pred_neu", "pred_pos"])
    print(cm_df.to_string())

    # save model and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr,    f"{MODELS_DIR}/logistic_regression.pkl")
    joblib.dump(tfidf, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    print(f"\n  ✓ Model saved to {MODELS_DIR}/logistic_regression.pkl")

    # save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = classification_report(y_test, y_pred,
                                   target_names=["negative", "neutral", "positive"],
                                   zero_division=0,
                                   output_dict=True)
    report["accuracy_overall"] = acc
    with open(f"{RESULTS_DIR}/lr_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return acc, lr, tfidf, X_test, y_test, y_pred


# ═══════════════════════════════════════════════════════
#  MODEL 2 — AfroXLMR TRANSFORMER
# ═══════════════════════════════════════════════════════

def train_afroxlmr(df: pd.DataFrame):
    print(f"\n{'─'*55}")
    print("  MODEL 2 — AfroXLMR Transformer")
    print(f"{'─'*55}")

    try:
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification,
                                  TrainingArguments,
                                  Trainer)
        from datasets import Dataset
        import torch
    except ImportError:
        print("  ✗ transformers/datasets not installed. Skipping.")
        return None

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print("  ⚠  Running on CPU — this will take 10–30 minutes.")
        print("     Consider running on Google Colab (free GPU) for speed.")

    # encode labels
    le = LabelEncoder()
    df = df.copy()
    df["label_id"] = le.fit_transform(df[LABEL_COL])
    label_names = list(le.classes_)
    print(f"  Label mapping: {dict(enumerate(label_names))}")

    # use raw text for transformer (handles noise better)
    text_col = TEXT_COL_TR if TEXT_COL_TR in df.columns else TEXT_COL_LR
    df[text_col] = df[text_col].astype(str)

    # train/test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df["label_id"]
    )
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # tokenizer
    MODEL_NAME = "Davlan/afro-xlmr-mini"   # lightweight version — faster
    print(f"\n  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch[text_col],
                         truncation=True,
                         padding="max_length",
                         max_length=128)

    # convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df[[text_col, "label_id"]].reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df[[text_col, "label_id"]].reset_index(drop=True))
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label_id", "labels")
    test_ds  = test_ds.rename_column("label_id", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    # load model
    print(f"  Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)}
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir=f"{MODELS_DIR}/afroxlmr",
        num_train_epochs=10,                   # more epochs for small dataset
        per_device_train_batch_size=8,         # smaller batch = more updates
        per_device_eval_batch_size=8,
        learning_rate=2e-5,                    # lower learning rate
        warmup_steps=100,                      # longer warmup
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",       # optimize for F1 not loss
        report_to="none"
    )

    # metrics function
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1": f1}

    from torch.nn import CrossEntropyLoss
    import torch

    # compute class weights to handle imbalance
    label_counts = df["label_id"].value_counts().sort_index()
    total = len(df)
    class_weights = torch.tensor(
        [total / (len(label_names) * label_counts[i]) for i in range(len(label_names))],
        dtype=torch.float
    )
    print(f"  Class weights: {class_weights.tolist()}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    # trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    print(f"\n  Training AfroXLMR (3 epochs)...")
    trainer.train()

    # evaluate
    print(f"\n  Evaluating...")
    preds_output = trainer.predict(test_ds)
    y_pred_ids   = np.argmax(preds_output.predictions, axis=-1)
    y_test_ids   = test_df["label_id"].values
    y_pred_names = le.inverse_transform(y_pred_ids)
    y_test_names = le.inverse_transform(y_test_ids)

    acc = accuracy_score(y_test_ids, y_pred_ids)
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_names, y_pred_names, zero_division=0))

    # save model
    trainer.save_model(f"{MODELS_DIR}/afroxlmr_final")
    tokenizer.save_pretrained(f"{MODELS_DIR}/afroxlmr_final")
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    print(f"\n  ✓ Model saved to {MODELS_DIR}/afroxlmr_final/")

    # save results
    report = classification_report(y_test_names, y_pred_names,
                                   zero_division=0, output_dict=True)
    report["accuracy_overall"] = acc
    with open(f"{RESULTS_DIR}/afroxlmr_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return acc


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Phase 4 — Model Training")
    print(f"{'='*55}\n")

    # load data
    df = load_data(INPUT_FILE)

    # model 1 — baseline
    lr_acc, lr_model, tfidf, X_test, y_test, y_pred = train_logistic_regression(df)

    # model 2 — transformer
    xlmr_acc = train_afroxlmr(df)

    # final comparison
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  Logistic Regression : {lr_acc*100:.1f}%")
    if xlmr_acc:
        print(f"  AfroXLMR            : {xlmr_acc*100:.1f}%")
        winner = "AfroXLMR" if xlmr_acc > lr_acc else "Logistic Regression"
        print(f"\n  🏆 Best model: {winner}")
    print(f"{'='*55}\n")
