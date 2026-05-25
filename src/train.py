"""
train.py
--------
Train a TF-IDF + Logistic Regression classifier on the synthesized notes dataset.

Lecture 3 (Preprocessing) techniques used:
- Train/test split (80/20)
- Normalization via TF-IDF
- Categorical label encoding

Lecture 4 (Methods) techniques used:
- Logistic Regression with sigmoid activation
- Text quantification with tf-idf and n-grams

Outputs:
- models/classifier.pkl  — trained pipeline
- models/metrics.json    — precision, recall, F1, accuracy per class
- models/confusion.png   — confusion matrix figure
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/notes.csv")
MODEL_DIR = Path("models")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna().drop_duplicates(subset=["text"])
    print(f"Loaded {len(df)} unique notes.")
    print("Label distribution:")
    print(df["label"].value_counts())
    return df


def build_pipeline() -> Pipeline:
    """TF-IDF (with bigrams) + LogisticRegression — straight from Lecture 4."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams (Lecture 4: n-grams)
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",  # handles any class imbalance
            random_state=42,
        )),
    ])


def plot_confusion_matrix(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X = df["text"].astype(str).tolist()
    y = df["label"].tolist()

    # Lecture 3: stratified split keeps class balance in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    pipe = build_pipeline()
    print("\nTraining...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    # Save artifacts
    joblib.dump(pipe, MODEL_DIR / "classifier.pkl")
    with (MODEL_DIR / "metrics.json").open("w") as f:
        json.dump({
            "accuracy": acc,
            "report": report,
            "labels": labels_sorted,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    plot_confusion_matrix(cm, labels_sorted, MODEL_DIR / "confusion.png")

    print(f"\nSaved: {MODEL_DIR/'classifier.pkl'}")
    print(f"Saved: {MODEL_DIR/'metrics.json'}")
    print(f"Saved: {MODEL_DIR/'confusion.png'}")


if __name__ == "__main__":
    main()
