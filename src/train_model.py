# #!/usr/bin/env python3
# from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABELS = ("REAL", "FAKE")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def plot_confusion(cm, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_curve(x, y, out, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")

    print("Loading data...")
    df_real = read_csv_any(Path(a.real))
    df_fake = read_csv_any(Path(a.fake))

    print(f"Loaded {len(df_real)} real, {len(df_fake)} fake rows")

    for df in [df_real, df_fake]:
        title = df["title"].fillna("") if "title" in df.columns else ""
        text = df.get(a.text_col, df.get("text", "")).fillna("")
        df["combined"] = (title + " " + text).str.strip()

    X = pd.concat([df_real["combined"], df_fake["combined"]], ignore_index=True)
    y = np.array([0] * len(df_real) + [1] * len(df_fake))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Training fast model (Logistic Regression)...")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    pipe.fit(X_train, y_train)

    print("Evaluating...")

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "report": classification_report(
            y_test, y_pred, target_names=LABELS, output_dict=True
        ),
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plot_curve(rec, prec, charts / "pr_curve.png", "PR Curve", "Recall", "Precision")

    joblib.dump(pipe, outdir / "pipeline.joblib")
    joblib.dump(pipe.named_steps["tfidf"], outdir / "vectorizer.joblib")
    joblib.dump(pipe.named_steps["clf"], outdir / "model.joblib")

    print("Done! Saved to:", outdir)


if __name__ == "__main__":
    main()
