"""
COMP70049 - Machine Learning in Cyber
=====================================
Section 1: Email Security & Phishing/Spam Detection
----------------------------------------------------
Detects malicious (spam/phishing) emails from their textual content.

Dataset : Enron Spam Dataset (33,716 labelled emails)
          Official source : https://www2.aueb.gr/users/ion/data/enron-spam/
          CSV compilation  : https://github.com/MWiechmann/enron_spam_data
          Expected local file: datasets/section1_email/enron_spam_data.csv

Models  : 1. Classic ML  -> TF-IDF features + Logistic Regression
          2. Deep Learning -> Word-embedding + LSTM network

Evaluation: Accuracy, Precision, Recall, F1-score, ROC curve/AUC,
            confusion matrices, training curves.

Usage   : python section1_phishing.py
Outputs : outputs/section1/  (figures + metrics CSV)

Author  : <student name>
"""

# --------------------------------------------------------------------------
# 1. Imports and configuration
# --------------------------------------------------------------------------
import os
import re
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ----------------------------- configuration ------------------------------
SEED          = 42                # global random seed for reproducibility
DATA_PATH     = os.path.join("datasets", "section1_email", "enron_spam_data.csv")
OUT_DIR       = os.path.join("outputs", "section1")
SAMPLE_SIZE   = 16000             # stratified subsample used for training
                                  # (set to None to use the full 33,716 emails)
VOCAB_SIZE    = 20000             # LSTM tokenizer vocabulary
MAX_LEN       = 200               # LSTM input length (tokens per email)
EMBED_DIM     = 64                # embedding dimension
LSTM_EPOCHS   = 6
LSTM_BATCH    = 64
TEST_FRACTION = 0.2

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# NLTK resources (stopword list + lemmatizer). Downloaded once, cached after.
import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS  = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# --------------------------------------------------------------------------
# 2. Data loading
# --------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load the Enron spam CSV and build a single text field per email."""
    df = pd.read_csv(path)
    # Subject + body give the model the complete textual context.
    df["text"]  = (df["Subject"].fillna("") + " " + df["Message"].fillna(""))
    df["label"] = (df["Spam/Ham"].str.lower() == "spam").astype(int)  # spam=1
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df[["text", "label"]]


# --------------------------------------------------------------------------
# 3. Text preprocessing
# --------------------------------------------------------------------------
URL_RE   = re.compile(r"http\S+|www\.\S+")
NUM_RE   = re.compile(r"\d+")
PUNCT_TB = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    """Lower-case, strip URLs/numbers/punctuation, remove stopwords,
    and lemmatize each remaining token."""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)      # keep a URL marker (phishing cue)
    text = NUM_RE.sub(" ", text)
    text = text.translate(PUNCT_TB)
    tokens = [LEMMATIZER.lemmatize(t) for t in text.split()
              if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# 4. Evaluation helpers
# --------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_score):
    """Compute the metric set required by the brief and return as dict."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {"Model":     name,
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall":    recall_score(y_true, y_pred),
            "F1-score":  f1_score(y_true, y_pred),
            "ROC-AUC":   auc(fpr, tpr),
            "_roc":      (fpr, tpr)}


def plot_confusion(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")


# --------------------------------------------------------------------------
# 5. Main experiment
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Section 1 - Phishing/Spam Email Detection")
    print("=" * 70)

    # ---------------- data ----------------
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df):,} emails "
          f"({df.label.sum():,} spam / {(1 - df.label).sum():,} ham)")

    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df = (df.groupby("label", group_keys=False)
                .sample(n=SAMPLE_SIZE // 2, random_state=SEED)
                .reset_index(drop=True))
        print(f"Stratified subsample: {len(df):,} emails (balanced)")

    print("Cleaning text (stopword removal + lemmatization)...")
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=TEST_FRACTION,
        stratify=df["label"], random_state=SEED)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ---------------- classic ML: TF-IDF + Logistic Regression -------------
    print("\n[1/2] TF-IDF + Logistic Regression")
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    Xtr_tfidf = vectorizer.fit_transform(X_train)
    Xte_tfidf = vectorizer.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, C=10.0, random_state=SEED)
    logreg.fit(Xtr_tfidf, y_train)

    lr_score = logreg.predict_proba(Xte_tfidf)[:, 1]
    lr_pred  = (lr_score >= 0.5).astype(int)
    res_lr   = evaluate("TF-IDF + Logistic Regression", y_test, lr_pred, lr_score)
    print(classification_report(y_test, lr_pred, target_names=["ham", "spam"]))

    # ---------------- deep learning: Embedding + LSTM ----------------------
    print("[2/2] Word-embedding + LSTM")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    # Pre-padding + masking: the LSTM then reads real tokens last and
    # ignores padding, which is essential for stable learning.
    Xtr_seq = pad_sequences(tokenizer.texts_to_sequences(X_train),
                            maxlen=MAX_LEN, padding="pre", truncating="post")
    Xte_seq = pad_sequences(tokenizer.texts_to_sequences(X_test),
                            maxlen=MAX_LEN, padding="pre", truncating="post")

    lstm = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    lstm.compile(optimizer="adam", loss="binary_crossentropy",
                 metrics=["accuracy"])
    lstm.summary()

    history = lstm.fit(Xtr_seq, y_train,
                       validation_split=0.1,
                       epochs=LSTM_EPOCHS,
                       batch_size=LSTM_BATCH,
                       callbacks=[EarlyStopping(patience=2,
                                                restore_best_weights=True)],
                       verbose=2)

    dl_score = lstm.predict(Xte_seq, verbose=0).ravel()
    dl_pred  = (dl_score >= 0.5).astype(int)
    res_dl   = evaluate("Embedding + LSTM", y_test, dl_pred, dl_score)
    print(classification_report(y_test, dl_pred, target_names=["ham", "spam"]))

    # ---------------- results table ----------------------------------------
    results = pd.DataFrame([{k: v for k, v in r.items() if k != "_roc"}
                            for r in (res_lr, res_dl)])
    results.to_csv(os.path.join(OUT_DIR, "section1_metrics.csv"), index=False)
    print("\nModel comparison:\n", results.round(4).to_string(index=False))

    # ---------------- visualisations ---------------------------------------
    # (a) confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_confusion(axes[0], y_test, lr_pred, "Logistic Regression")
    plot_confusion(axes[1], y_test, dl_pred, "LSTM")
    fig.suptitle("Section 1 - Confusion Matrices (Enron Spam test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s1_confusion_matrices.png"), dpi=150)

    # (b) ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in (res_lr, res_dl):
        fpr, tpr = r["_roc"]
        ax.plot(fpr, tpr, label=f"{r['Model']} (AUC = {r['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Section 1 - ROC Curves"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s1_roc_curves.png"), dpi=150)

    # (c) LSTM training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("LSTM loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("LSTM accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.suptitle("Section 1 - LSTM Training Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s1_lstm_training.png"), dpi=150)

    # (d) most indicative TF-IDF features (model interpretability)
    feat = np.array(vectorizer.get_feature_names_out())
    coef = logreg.coef_.ravel()
    top  = np.argsort(coef)
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.concatenate([top[:10], top[-10:]])
    colors = ["#2b8cbe"] * 10 + ["#e34a33"] * 10
    ax.barh(range(20), coef[idx], color=colors)
    ax.set_yticks(range(20)); ax.set_yticklabels(feat[idx])
    ax.set_title("Top ham (blue) vs spam (red) indicators - LogReg weights")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s1_top_features.png"), dpi=150)

    print(f"\nAll figures and metrics written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
