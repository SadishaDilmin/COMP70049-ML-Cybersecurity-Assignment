# COMP70049 — Machine Learning in Cyber: Assignment Submission

**Student ID:** CB018287
**Repository:** https://github.com/SadishaDilmin/COMP70049-ML-Cybersecurity-Assignment

This single README covers **all four sections** of the assignment: how to set
the project up, how to run each section, and what each run produces.

---

## 0. Quick Start (for the marker)

Everything needed to reproduce the results — code, datasets and the written
report — is committed to this repository. There is nothing to download
separately.

```bash
# 1. Clone the repository (≈190 MB on disk — the datasets are included)
git clone https://github.com/SadishaDilmin/COMP70049-ML-Cybersecurity-Assignment.git
cd COMP70049-ML-Cybersecurity-Assignment

# 2. Create and activate a virtual environment (Python 3.10–3.12)
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run any section from the project root
python section1_phishing.py
```

Total runtime for all four sections is roughly **20 minutes on CPU** — no GPU
is required. Each script prints its results to the console and writes figures
and metrics into `outputs/sectionN/`.

> **Just want to see the results without running anything?**
> The `outputs/` folders in this repository already contain every figure,
> metrics CSV and console log (`run_log.txt`) from the exact runs reported in
> `Report.docx`. `PROJECT_OVERVIEW.md` gives a plain-language walkthrough of
> what each section does and what it found.

---

## 1. Section Identifiers

| Section | Topic | Script |
|---|---|---|
| Section 1 | Email Security & Phishing Detection | `section1_phishing.py` |
| Section 2 | Cyber Attack Detection (network intrusion classification) | `section2_intrusion.py` |
| Section 3 | Anomaly Detection for Cyber Threats | `section3_anomaly.py` |
| Section 4 | Ransomware Detection & Prevention | `section4_ransomware.py` |

Each section is implemented in its own standalone Python file, as required.

---

## 2. Folder Structure

```
COMP70049-ML-Cybersecurity-Assignment/
├── section1_phishing.py            # Section 1: TF-IDF + LogReg vs Embedding + LSTM
├── section2_intrusion.py           # Section 2: Random Forest vs 1-D CNN (5-class)
├── section3_anomaly.py             # Section 3: Isolation Forest vs Autoencoder
├── section4_ransomware.py          # Section 4: Gradient Boosting vs LSTM (windows)
├── requirements.txt                # Python dependencies
├── README.md                       # This file — setup and run instructions
├── PROJECT_OVERVIEW.md             # Plain-language summary of all four sections
├── Report.docx                     # Written report (≤3000 words)
├── datasets/
│   ├── section1_email/enron_spam_data.csv          # Enron Spam Dataset (33,716 emails)
│   ├── section2_nslkdd/KDDTrain+.txt               # NSL-KDD training split
│   ├── section2_nslkdd/KDDTest+.txt                # NSL-KDD official test split
│   ├── section3_unsw/UNSW_NB15_training-set.csv    # UNSW-NB15 (see note below)
│   ├── section3_unsw/UNSW_NB15_testing-set.csv
│   └── section4_ransomware/Ransomware_Data.csv     # CSU Sysmon ransomware dataset
└── outputs/
    ├── section1/   # figures (.png), metrics (.csv) and run log for Section 1
    ├── section2/   # ... Section 2
    ├── section3/   # ... Section 3
    └── section4/   # ... Section 4
```

The exact dataset files used by the implementation are included in
`datasets/` (all are public datasets; official sources in section 8 below).

---

## 3. System Requirements

- **Python:** 3.10 – 3.12 (developed and tested on Python 3.11).
  Python 3.13 is **not** recommended — TensorFlow support lags behind.
- **OS:** Windows, macOS or Linux.
- **Hardware:** no GPU required — every model trains on CPU.
  Approximate CPU runtimes: Section 1 ≈ 6 min, Section 2 ≈ 6 min,
  Section 3 ≈ 3 min, Section 4 ≈ 5 min.
- **RAM:** 8 GB recommended.
- **Disk:** ≈190 MB for the cloned repository, plus ≈600 MB for the virtual
  environment (TensorFlow is a large dependency).
- **Internet:** needed once, for `pip install` and for Section 1's small NLTK
  downloads. The scripts run offline afterwards.
- The scripts also run unchanged in **Google Colab** (upload the project
  folder, `pip install -r requirements.txt`, then run each script).

---

## 4. Required Libraries

Listed in `requirements.txt`: `numpy`, `pandas`, `scikit-learn`,
`matplotlib`, `seaborn`, `nltk`, `tensorflow` (CPU build is sufficient).

Section 1 additionally downloads three small NLTK resources on first run
(`stopwords`, `wordnet`, `omw-1.4`); this happens automatically and is
cached afterwards.

---

## 5. Installation Instructions

```bash
# (optional but recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# install all dependencies
pip install -r requirements.txt
```

To confirm the install worked before running a full section:

```bash
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, nltk, tensorflow; print('All dependencies OK')"
```

---

## 6. How to Run the Code

Run each section **from the project root** (the folder containing this
README), so the relative `datasets/` and `outputs/` paths resolve:

```bash
python section1_phishing.py     # Section 1 — phishing email detection
python section2_intrusion.py    # Section 2 — intrusion classification
python section3_anomaly.py      # Section 3 — anomaly detection
python section4_ransomware.py   # Section 4 — ransomware detection
```

The four sections are completely independent and can be run in any order, or
individually. Re-running a section overwrites that section's files in
`outputs/sectionN/`.

Each script is fully self-contained: it loads its dataset from
`datasets/`, performs preprocessing, trains the classic ML model and the
deep learning model, evaluates both, prints a comparison table, and
saves all figures/metrics into its `outputs/sectionN/` folder.
A fixed random seed (42) is set everywhere for reproducibility
(deep-learning numbers may still vary marginally between platforms).

---

## 7. Expected Outputs

Each run prints per-class classification reports and a final model
comparison table to the console, and writes to `outputs/sectionN/`:

| Section | Files produced |
|---|---|
| 1 | `section1_metrics.csv`, `s1_confusion_matrices.png`, `s1_roc_curves.png`, `s1_lstm_training.png`, `s1_top_features.png` |
| 2 | `section2_metrics.csv`, `s2_confusion_matrices.png`, `s2_pr_curves.png`, `s2_cnn_training.png`, `s2_rf_importances.png`, `s2_pca_variance.png` |
| 3 | `section3_metrics.csv`, `s3_confusion_matrices.png`, `s3_roc_pr_curves.png`, `s3_reconstruction_error.png`, `s3_ae_training.png` |
| 4 | `section4_metrics.csv`, `s4_confusion_matrices.png`, `s4_roc_pr_curves.png`, `s4_lstm_training.png`, `s4_feature_importance.png` |

The committed `outputs/` folders already contain the artefacts from the
runs reported in `Report.docx`, including each run's console log
(`run_log.txt`). These can be compared directly against a fresh run.

Headline results from those runs (full discussion in `Report.docx`):

| Section | Classic ML model | Deep learning model |
|---|---|---|
| 1 — Phishing | TF-IDF + Logistic Regression — 99.1% | Embedding + LSTM — 98.7% |
| 2 — Intrusion | Random Forest — 75.6% | 1-D CNN — 75.2% |
| 3 — Anomaly | Isolation Forest — 67.4% | Autoencoder — 82.0% |
| 4 — Ransomware | Gradient Boosting — 95.8% | LSTM (windows) — 99.9% |

---

## 8. Datasets: Sources and Modifications

| Section | Dataset | Official source | Download used |
|---|---|---|---|
| 1 | Enron Spam Dataset | https://www2.aueb.gr/users/ion/data/enron-spam/ | CSV compilation: https://github.com/MWiechmann/enron_spam_data |
| 2 | NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | Mirror: https://github.com/defcom17/NSL_KDD |
| 3 | UNSW-NB15 (train/test partitions) | https://research.unsw.edu.au/projects/unsw-nb15-dataset | Mirror: https://github.com/InitRoot/UNSW_NB15 |
| 4 | CSU Ransomware Behavioural Dataset (Sysmon events) | https://github.com/CSCRC-SCREED/CSU-Ransomware-Data | same |

All four are public, non-sensitive research datasets — no real client,
personal or proprietary data is used anywhere in this project.

**Modifications before implementation:** none — the files in `datasets/`
are exactly as downloaded (Section 3's mirror ships the categorical
columns already integer-coded as `xProt`/`xServ`/`xState`, and its two
CSVs are name-swapped relative to the official splits; the script
detects and corrects the swap by row count at load time). All cleaning,
encoding, scaling and feature engineering is performed inside the
scripts at runtime, as documented in the report.

---

## 9. Notes / Assumptions

- **Section 1** trains on a balanced, stratified subsample of 16,000
  emails (of 33,716) to keep LSTM training practical on CPU; set
  `SAMPLE_SIZE = None` at the top of the script to use the full corpus.
- **Section 2** evaluates on the official `KDDTest+` split, which
  contains attack types never seen in training — accuracies of ~75-80%
  are the expected, well-documented behaviour for this benchmark (see
  discussion in the report).
- **Section 3** follows a strict unsupervised protocol: both detectors
  are fitted on normal training traffic only; attack labels are used
  exclusively for evaluation. Duplicates are removed from training data
  only; the anomaly-score threshold is set to the 95th percentile of
  reconstruction error on normal training data.
- **Section 4** treats each Sysmon event row as one behavioural
  observation. Duplicate events are removed so per-event train/test
  splits share no identical rows; the LSTM classifies windows of 10
  consecutive events (stride 5) with a chronological 80/20 split per
  class to prevent overlapping windows leaking between train and test.
- All figures are saved with a headless-safe Matplotlib backend, so the
  scripts run on servers/Colab without a display.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: datasets/...` | The script was not run from the project root. `cd` into the folder containing this README, then run `python section1_phishing.py`. |
| `pip install` fails on `tensorflow` | Python version is too new (3.13+) or too old. Use Python 3.10–3.12. Check with `python3 --version`. |
| `nltk` download errors on Section 1 | No internet on first run. Connect once and re-run — the resources are cached afterwards and later runs work offline. |
| TensorFlow prints `oneDNN` / CUDA / retracing warnings | Harmless informational messages on CPU. Training proceeds normally. |
| Deep-learning numbers differ slightly from the report | Expected. Neural network training is not bit-identical across platforms even with a fixed seed; classic ML results reproduce exactly. |
| `python` not found (macOS/Linux) | Use `python3` instead, or activate the virtual environment first. |

---

## 11. Contact Information

Student ID: CB018287
Student email: CB018287@students.apiit.lk
