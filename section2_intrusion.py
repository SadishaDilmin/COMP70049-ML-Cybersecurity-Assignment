"""
COMP70049 - Machine Learning in Cyber
=====================================
Section 2: Cyber Attack Detection (Network Intrusion Classification)
---------------------------------------------------------------------
Classifies network connections into five classes:
Normal, DoS, Probe, R2L (remote-to-local), U2R (user-to-root).

Dataset : NSL-KDD (improved KDD'99), official page:
          https://www.unb.ca/cic/datasets/nsl.html
          Mirror used for download: https://github.com/defcom17/NSL_KDD
          Expected local files:
            datasets/section2_nslkdd/KDDTrain+.txt
            datasets/section2_nslkdd/KDDTest+.txt

Models  : 1. Classic ML   -> Random Forest
          2. Deep Learning -> 1-D Convolutional Neural Network (CNN)

Evaluation: Accuracy, macro precision/recall/F1, confusion matrices,
            per-class precision-recall curves, PCA variance analysis.

Usage   : python section2_intrusion.py
Outputs : outputs/section2/
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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

SEED       = 42
DATA_DIR   = os.path.join("datasets", "section2_nslkdd")
OUT_DIR    = os.path.join("outputs", "section2")
CNN_EPOCHS = 8
CNN_BATCH  = 256

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# NSL-KDD column names (41 features + label + difficulty), per dataset docs.
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "difficulty",
]

# Mapping of the 39 fine-grained attack names to 4 attack categories.
ATTACK_MAP = {
    # DoS
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


# --------------------------------------------------------------------------
# 2. Data loading and preprocessing
# --------------------------------------------------------------------------
def load_split(fname: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, fname), names=COLUMNS)
    df["category"] = df["attack"].map(ATTACK_MAP).fillna("Normal")
    return df


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot encode categoricals, drop constant column, standard-scale.
    Encoders/scalers are fitted ONLY on training data (no test leakage)."""
    y_train = train["category"].values
    y_test  = test["category"].values

    X_train = train.drop(columns=["attack", "difficulty", "category"])
    X_test  = test.drop(columns=["attack", "difficulty", "category"])

    # num_outbound_cmds is constant (all zeros) -> carries no information.
    X_train = X_train.drop(columns=["num_outbound_cmds"])
    X_test  = X_test.drop(columns=["num_outbound_cmds"])

    # One-hot encode the three categorical features; align test to train.
    cats = ["protocol_type", "service", "flag"]
    X_train = pd.get_dummies(X_train, columns=cats)
    X_test  = pd.get_dummies(X_test, columns=cats)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(float))
    X_test  = scaler.transform(X_test.astype(float))

    le = LabelEncoder().fit(CLASS_ORDER)
    return (X_train, le.transform(y_train),
            X_test,  le.transform(y_test), le)


# --------------------------------------------------------------------------
# 3. Main experiment
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Section 2 - Network Intrusion Classification (NSL-KDD)")
    print("=" * 70)

    train = load_split("KDDTrain+.txt")
    test  = load_split("KDDTest+.txt")
    print(f"Train: {len(train):,} records   Test: {len(test):,} records")
    print("Train class distribution:\n", train["category"].value_counts())

    X_train, y_train, X_test, y_test, le = preprocess(train, test)
    n_features = X_train.shape[1]
    print(f"Feature space after one-hot encoding + scaling: {n_features}")

    # ------------- PCA analysis (dimensionality insight, per brief) --------
    pca = PCA(random_state=SEED).fit(X_train)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    print(f"PCA: {n95} components explain 95% of the variance")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cum[:80], lw=2)
    ax.axhline(0.95, color="r", ls="--", label="95% variance")
    ax.axvline(n95, color="g", ls="--", label=f"{n95} components")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Section 2 - PCA scree analysis (NSL-KDD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_pca_variance.png"), dpi=150)

    # ---------------- classic ML: Random Forest ----------------------------
    print("\n[1/2] Random Forest (full one-hot feature space)")
    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                random_state=SEED)
    rf.fit(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    print(classification_report(y_test, rf_pred,
                                target_names=le.classes_, digits=4))

    # ---------------- deep learning: 1-D CNN -------------------------------
    print("[2/2] 1-D Convolutional Neural Network")
    Xtr_cnn = X_train[..., np.newaxis]        # (samples, features, 1)
    Xte_cnn = X_test[..., np.newaxis]

    cnn = Sequential([
        Input(shape=(n_features, 1)),
        Conv1D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(len(le.classes_), activation="softmax"),
    ])
    cnn.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    cnn.summary()

    history = cnn.fit(Xtr_cnn, y_train,
                      validation_split=0.1,
                      epochs=CNN_EPOCHS, batch_size=CNN_BATCH,
                      callbacks=[EarlyStopping(patience=2,
                                               restore_best_weights=True)],
                      verbose=2)

    cnn_proba = cnn.predict(Xte_cnn, verbose=0)
    cnn_pred  = cnn_proba.argmax(axis=1)
    print(classification_report(y_test, cnn_pred,
                                target_names=le.classes_, digits=4))

    # ---------------- results table ----------------------------------------
    rows = []
    for name, pred in [("Random Forest", rf_pred), ("1-D CNN", cnn_pred)]:
        rows.append({
            "Model": name,
            "Accuracy":        accuracy_score(y_test, pred),
            "Macro Precision": precision_score(y_test, pred, average="macro"),
            "Macro Recall":    recall_score(y_test, pred, average="macro"),
            "Macro F1":        f1_score(y_test, pred, average="macro"),
            "Weighted F1":     f1_score(y_test, pred, average="weighted"),
        })
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT_DIR, "section2_metrics.csv"), index=False)
    print("\nModel comparison:\n", results.round(4).to_string(index=False))

    # ---------------- visualisations ---------------------------------------
    # (a) confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, pred) in zip(axes, [("Random Forest", rf_pred),
                                       ("1-D CNN", cnn_pred)]):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Section 2 - Confusion Matrices (KDDTest+)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_confusion_matrices.png"), dpi=150)

    # (b) per-class precision-recall curves (one-vs-rest)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, proba) in zip(axes, [("Random Forest", rf_proba),
                                        ("1-D CNN", cnn_proba)]):
        for k, cls in enumerate(le.classes_):
            y_bin = (y_test == k).astype(int)
            p, r, _ = precision_recall_curve(y_bin, proba[:, k])
            ap = average_precision_score(y_bin, proba[:, k])
            ax.plot(r, p, label=f"{cls} (AP={ap:.3f})")
        ax.set_title(name); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
    fig.suptitle("Section 2 - Precision-Recall Curves (one-vs-rest)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_pr_curves.png"), dpi=150)

    # (c) CNN training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("CNN loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("CNN accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.suptitle("Section 2 - CNN Training Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_cnn_training.png"), dpi=150)

    # (d) Random Forest feature importances (top 20)
    # NOTE: importances refer to the one-hot encoded feature space.
    fig, ax = plt.subplots(figsize=(8, 6))
    # Recover feature names by re-running the dummy encoding on column names
    train_cols = (pd.get_dummies(
        load_split("KDDTrain+.txt")
        .drop(columns=["attack", "difficulty", "category", "num_outbound_cmds"]),
        columns=["protocol_type", "service", "flag"]).columns)
    imp = pd.Series(rf.feature_importances_, index=train_cols).nlargest(20)
    imp.iloc[::-1].plot.barh(ax=ax, color="#2b8cbe")
    ax.set_title("Section 2 - Random Forest top-20 feature importances")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_rf_importances.png"), dpi=150)

    print(f"\nAll figures and metrics written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
