"""
COMP70049 - Machine Learning in Cyber
=====================================
Section 3: Anomaly Detection for Cyber Threats
-----------------------------------------------
Unsupervised detection of anomalous (attack) network traffic: both models
are trained ONLY on normal traffic; attack labels are used solely for
evaluation, mimicking a zero-day detection scenario.

Dataset : UNSW-NB15, official page:
          https://research.unsw.edu.au/projects/unsw-nb15-dataset
          Mirror used for download: https://github.com/InitRoot/UNSW_NB15
          Expected local files:
            datasets/section3_unsw/UNSW_NB15_training-set.csv
            datasets/section3_unsw/UNSW_NB15_testing-set.csv
          (In this mirror the categorical columns proto/service/state are
           already integer-coded as xProt/xServ/xState.)

Models  : 1. Classic ML   -> Isolation Forest
          2. Deep Learning -> Dense Autoencoder (reconstruction error)

Evaluation: TPR, FPR, Precision/Recall/F1, ROC + Precision-Recall curves,
            reconstruction-error distribution.

Usage   : python section3_anomaly.py
Outputs : outputs/section3/
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc, precision_recall_curve,
                             average_precision_score, accuracy_score)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

SEED       = 42
DATA_DIR   = os.path.join("datasets", "section3_unsw")
OUT_DIR    = os.path.join("outputs", "section3")
AE_EPOCHS  = 15
AE_BATCH   = 256
NORMAL_TRAIN_CAP = 50000     # cap on normal records used for training
CONTAM     = 0.05            # Isolation Forest contamination assumption

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 2. Data loading and preprocessing
# --------------------------------------------------------------------------
def load_data():
    """Load UNSW-NB15 train/test CSVs and produce scaled feature matrices.

    Preprocessing steps (per the brief):
      - drop identifier column and duplicates
      - median-impute any missing values
      - log1p-transform the features: traffic statistics (bytes, rates,
        loads, jitter...) are extremely heavy-tailed, and compressing
        them makes normal/attack structure far easier to model
      - Min-Max normalise to [0, 1], fitted on ONLY normal training
        traffic (unsupervised protocol; autoencoders train best on
        bounded inputs). Test values are clipped to [0, 1] so unseen
        extremes cannot dominate the reconstruction error.
    """
    train = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"))

    # The official UNSW-NB15 training split has 175,341 records and the
    # testing split 82,332. Some mirrors ship the two files with swapped
    # names, so assign the splits by size to be robust.
    if len(train) < len(test):
        print("NOTE: swapped train/test files detected by size - correcting.")
        train, test = test, train

    train = train.drop(columns=["id"], errors="ignore")
    test  = test.drop(columns=["id"], errors="ignore")

    # Remove duplicate records from the TRAINING data only, so repeated
    # identical flows cannot bias what the models learn as "normal".
    # The test split is evaluated as distributed (deployment conditions).
    train = train.drop_duplicates().reset_index(drop=True)

    y_train = train.pop("label").values      # 0 = normal, 1 = attack
    y_test  = test.pop("label").values

    # median imputation (defensive; this export contains no missing values)
    train = train.fillna(train.median(numeric_only=True))
    test  = test.fillna(train.median(numeric_only=True))

    # compress heavy-tailed magnitudes
    train = np.log1p(train.clip(lower=0))
    test  = np.log1p(test.clip(lower=0))

    # fit the scaler on NORMAL TRAINING data only (unsupervised protocol)
    scaler  = MinMaxScaler().fit(train[y_train == 0])
    X_train = np.clip(scaler.transform(train), 0, 1)
    X_test  = np.clip(scaler.transform(test), 0, 1)
    return X_train, y_train, X_test, y_test, train.columns


def tpr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn), fp / (fp + tn)


# --------------------------------------------------------------------------
# 3. Main experiment
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Section 3 - Anomaly Detection (UNSW-NB15)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, feat_names = load_data()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Test set: {int(y_test.sum()):,} attacks / "
          f"{int((1 - y_test).sum()):,} normal")

    # Both detectors learn from NORMAL training traffic only.
    X_norm = X_train[y_train == 0]
    if len(X_norm) > NORMAL_TRAIN_CAP:
        idx = np.random.default_rng(SEED).choice(
            len(X_norm), NORMAL_TRAIN_CAP, replace=False)
        X_norm = X_norm[idx]
    print(f"Normal-only training records: {len(X_norm):,}")

    # ---------------- classic ML: Isolation Forest -------------------------
    print("\n[1/2] Isolation Forest")
    iso = IsolationForest(n_estimators=200, contamination=CONTAM,
                          random_state=SEED, n_jobs=-1)
    iso.fit(X_norm)
    # decision_function: high = normal. Negate so high score = anomalous.
    iso_score = -iso.decision_function(X_test)
    iso_pred  = (iso.predict(X_test) == -1).astype(int)

    # ---------------- deep learning: Autoencoder ---------------------------
    print("[2/2] Dense Autoencoder")
    n_feat = X_train.shape[1]
    inp = Input(shape=(n_feat,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    z = Dense(16, activation="relu", name="bottleneck")(x)
    x = Dense(32, activation="relu")(z)
    x = Dense(64, activation="relu")(x)
    out = Dense(n_feat, activation="sigmoid")(x)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.summary()

    history = ae.fit(X_norm, X_norm,
                     validation_split=0.1,
                     epochs=AE_EPOCHS, batch_size=AE_BATCH,
                     callbacks=[EarlyStopping(patience=3,
                                              restore_best_weights=True)],
                     verbose=2)

    # anomaly score = per-record reconstruction MSE
    recon    = ae.predict(X_test, verbose=0)
    ae_score = np.mean((X_test - recon) ** 2, axis=1)

    # threshold = 95th percentile of reconstruction error on normal
    # training traffic (same 5% "budget" as the Isolation Forest).
    val_recon  = ae.predict(X_norm, verbose=0)
    val_err    = np.mean((X_norm - val_recon) ** 2, axis=1)
    threshold  = np.percentile(val_err, 100 * (1 - CONTAM))
    ae_pred    = (ae_score > threshold).astype(int)
    print(f"Autoencoder threshold (95th pct of normal error): {threshold:.6f}")

    # ---------------- metrics ----------------------------------------------
    rows = []
    for name, pred, score in [("Isolation Forest", iso_pred, iso_score),
                              ("Autoencoder",      ae_pred,  ae_score)]:
        tpr, fpr = tpr_fpr(y_test, pred)
        f, t, _ = roc_curve(y_test, score)
        rows.append({
            "Model":     name,
            "TPR (Detection rate)": tpr,
            "FPR (False alarms)":   fpr,
            "Accuracy":  accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall":    recall_score(y_test, pred),
            "F1-score":  f1_score(y_test, pred),
            "ROC-AUC":   auc(f, t),
            "PR-AUC":    average_precision_score(y_test, score),
        })
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT_DIR, "section3_metrics.csv"), index=False)
    print("\nModel comparison:\n", results.round(4).to_string(index=False))

    # ---------------- visualisations ---------------------------------------
    # (a) ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, score in [("Isolation Forest", iso_score),
                        ("Autoencoder", ae_score)]:
        f, t, _ = roc_curve(y_test, score)
        axes[0].plot(f, t, label=f"{name} (AUC={auc(f, t):.3f})")
        p, r, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        axes[1].plot(r, p, label=f"{name} (AP={ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC curves"); axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].set_title("Precision-Recall curves")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()
    fig.suptitle("Section 3 - Anomaly Detection Performance (UNSW-NB15)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s3_roc_pr_curves.png"), dpi=150)

    # (b) reconstruction-error distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, np.percentile(ae_score, 99), 80)
    ax.hist(ae_score[y_test == 0], bins=bins, alpha=0.6, density=True,
            label="Normal traffic", color="#2b8cbe")
    ax.hist(ae_score[y_test == 1], bins=bins, alpha=0.6, density=True,
            label="Attack traffic", color="#e34a33")
    ax.axvline(threshold, color="k", ls="--", label="Detection threshold")
    ax.set_xlabel("Autoencoder reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Section 3 - Reconstruction Error: Normal vs Attack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s3_reconstruction_error.png"), dpi=150)

    # (c) confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, pred) in zip(axes, [("Isolation Forest", iso_pred),
                                       ("Autoencoder", ae_pred)]):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Section 3 - Confusion Matrices (UNSW-NB15 test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s3_confusion_matrices.png"), dpi=150)

    # (d) autoencoder training curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("Section 3 - Autoencoder Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s3_ae_training.png"), dpi=150)

    print(f"\nAll figures and metrics written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
