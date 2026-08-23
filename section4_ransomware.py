"""
COMP70049 - Machine Learning in Cyber
=====================================
Section 4: Ransomware Detection & Prevention
---------------------------------------------
Detects ransomware activity from host behaviour telemetry (Sysmon events:
file creation/deletion, registry changes, process creation, etc.).

Dataset : CSU Ransomware Behavioural Dataset (CSCRC-SCREED)
          352,876 Sysmon events (271,993 goodware / 80,883 ransomware)
          collected in a controlled sandbox for multiple modern families
          (LockBit, Medusa, Akira, BlackBasta, ...).
          Source: https://github.com/CSCRC-SCREED/CSU-Ransomware-Data
          Expected local file:
            datasets/section4_ransomware/Ransomware_Data.csv

Models  : 1. Classic ML   -> (Histogram) Gradient Boosting Classifier
                             on individual behavioural events
          2. Deep Learning -> LSTM over sliding windows of consecutive
                             events, modelling system behaviour OVER TIME

Evaluation: Precision, Recall, F1-score (per the brief), plus ROC/PR
            curves, confusion matrices and feature importance.

Usage   : python section4_ransomware.py
Outputs : outputs/section4/
"""

# --------------------------------------------------------------------------
# 1. Imports and configuration
# --------------------------------------------------------------------------
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

SEED        = 42
DATA_PATH   = os.path.join("datasets", "section4_ransomware",
                           "Ransomware_Data.csv")
OUT_DIR     = os.path.join("outputs", "section4")
WINDOW      = 10          # events per LSTM window
STRIDE      = 5           # window step (50% overlap)
MAX_EVENTS_PER_CLASS = 120000   # cap for LSTM sequence building
LSTM_EPOCHS = 6
LSTM_BATCH  = 128

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 2. Data loading and preprocessing
# --------------------------------------------------------------------------
def load_data():
    """Load the Sysmon event dataset.

    Preprocessing:
      - map the label ('good'/'ransom') to 0/1
      - drop duplicate events
      - median-impute missing numeric values (defensive)
      - standard-scale features (fitted on training data only, later)
    """
    df = pd.read_csv(DATA_PATH)
    df["label"] = (df["Ware Type"].str.strip().str.lower()
                   .eq("ransom").astype(int))
    df = df.drop(columns=["Ware Type"])
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n0 - len(df):,} duplicate events")
    df = df.fillna(df.median(numeric_only=True))
    return df


def build_windows(X, y, window=WINDOW, stride=STRIDE):
    """Slice the (time-ordered) event stream of each class into overlapping
    windows so the LSTM can learn the temporal signature of ransomware
    behaviour (e.g. bursts of file writes + deletions + registry changes).

    Windows are built within a class segment only, so every window has an
    unambiguous label."""
    seqs, labels = [], []
    for cls in (0, 1):
        Xc = X[y == cls]
        if len(Xc) > MAX_EVENTS_PER_CLASS:
            Xc = Xc[:MAX_EVENTS_PER_CLASS]
        for start in range(0, len(Xc) - window + 1, stride):
            seqs.append(Xc[start:start + window])
            labels.append(cls)
    return np.asarray(seqs, dtype=np.float32), np.asarray(labels)


# --------------------------------------------------------------------------
# 3. Main experiment
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Section 4 - Ransomware Detection (Sysmon behavioural events)")
    print("=" * 70)

    df = load_data()
    y = df.pop("label").values
    feat_names = df.columns.tolist()
    print(f"Events: {len(df):,}  "
          f"({int(y.sum()):,} ransomware / {int((1 - y).sum()):,} goodware)")
    print(f"Features ({len(feat_names)}): {feat_names}")

    # ---------------- classic ML: Gradient Boosting on single events -------
    print("\n[1/2] Histogram Gradient Boosting Classifier (per-event)")
    X_train, X_test, y_train, y_test = train_test_split(
        df.values, y, test_size=0.2, stratify=y, random_state=SEED)

    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         random_state=SEED)
    gbm.fit(X_train, y_train)
    gbm_score = gbm.predict_proba(X_test)[:, 1]
    gbm_pred  = (gbm_score >= 0.5).astype(int)
    print(classification_report(y_test, gbm_pred,
                                target_names=["goodware", "ransomware"],
                                digits=4))

    # ---------------- deep learning: LSTM over event windows ---------------
    print("[2/2] LSTM over windows of consecutive events "
          f"(window={WINDOW}, stride={STRIDE})")
    scaler_seq = StandardScaler().fit(df.values)   # scale before windowing
    X_seq, y_seq = build_windows(scaler_seq.transform(df.values), y)
    print(f"Windows built: {len(X_seq):,} "
          f"({int(y_seq.sum()):,} ransomware / {int((1-y_seq).sum()):,} goodware)")

    # Chronological split per class: first 80% train, last 20% test.
    # (A random split would let overlapping windows leak between sets.)
    tr_idx, te_idx = [], []
    for cls in (0, 1):
        idx = np.where(y_seq == cls)[0]
        cut = int(0.8 * len(idx))
        tr_idx.extend(idx[:cut]); te_idx.extend(idx[cut:])
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    Xs_tr, ys_tr = X_seq[tr_idx], y_seq[tr_idx]
    Xs_te, ys_te = X_seq[te_idx], y_seq[te_idx]
    print(f"Sequence train: {len(Xs_tr):,}  test: {len(Xs_te):,}")

    lstm = Sequential([
        Input(shape=(WINDOW, X_seq.shape[2])),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    lstm.compile(optimizer="adam", loss="binary_crossentropy",
                 metrics=["accuracy"])
    lstm.summary()

    history = lstm.fit(Xs_tr, ys_tr,
                       validation_split=0.1,
                       epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                       callbacks=[EarlyStopping(patience=2,
                                                restore_best_weights=True)],
                       verbose=2)

    lstm_score = lstm.predict(Xs_te, verbose=0).ravel()
    lstm_pred  = (lstm_score >= 0.5).astype(int)
    print(classification_report(ys_te, lstm_pred,
                                target_names=["goodware", "ransomware"],
                                digits=4))

    # ---------------- results table ----------------------------------------
    rows = []
    for name, yt, pred, score in [
            ("Gradient Boosting (per event)", y_test, gbm_pred, gbm_score),
            ("LSTM (event windows)",          ys_te,  lstm_pred, lstm_score)]:
        f, t, _ = roc_curve(yt, score)
        rows.append({
            "Model":     name,
            "Accuracy":  accuracy_score(yt, pred),
            "Precision": precision_score(yt, pred),
            "Recall":    recall_score(yt, pred),
            "F1-score":  f1_score(yt, pred),
            "ROC-AUC":   auc(f, t),
            "PR-AUC":    average_precision_score(yt, score),
        })
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT_DIR, "section4_metrics.csv"), index=False)
    print("\nModel comparison:\n", results.round(4).to_string(index=False))

    # ---------------- visualisations ---------------------------------------
    # (a) confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, yt, pred) in zip(axes, [
            ("Gradient Boosting", y_test, gbm_pred),
            ("LSTM", ys_te, lstm_pred)]):
        cm = confusion_matrix(yt, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Goodware", "Ransomware"],
                    yticklabels=["Goodware", "Ransomware"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Section 4 - Confusion Matrices")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s4_confusion_matrices.png"), dpi=150)

    # (b) ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, yt, score in [("Gradient Boosting", y_test, gbm_score),
                            ("LSTM", ys_te, lstm_score)]:
        f, t, _ = roc_curve(yt, score)
        axes[0].plot(f, t, label=f"{name} (AUC={auc(f, t):.4f})")
        p, r, _ = precision_recall_curve(yt, score)
        axes[1].plot(r, p,
                     label=f"{name} (AP={average_precision_score(yt, score):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC curves"); axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].set_title("Precision-Recall curves")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()
    fig.suptitle("Section 4 - Ransomware Detection Performance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s4_roc_pr_curves.png"), dpi=150)

    # (c) LSTM training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("LSTM loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("LSTM accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle("Section 4 - LSTM Training Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s4_lstm_training.png"), dpi=150)

    # (d) permutation feature importance for the GBM (on a test subsample)
    print("Computing permutation feature importance (subsample)...")
    n_sub = min(8000, len(X_test))
    sub = np.random.default_rng(SEED).choice(len(X_test), n_sub, replace=False)
    imp = permutation_importance(gbm, X_test[sub], y_test[sub],
                                 n_repeats=5, random_state=SEED, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feat_names)[order], imp.importances_mean[order],
            color="#2b8cbe")
    ax.set_title("Section 4 - GBM permutation feature importance")
    ax.set_xlabel("Mean accuracy drop when permuted")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s4_feature_importance.png"), dpi=150)

    print(f"\nAll figures and metrics written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
