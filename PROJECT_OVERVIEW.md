# Project Overview — COMP70049 Machine Learning in Cyber

A plain-language walkthrough of what each section does, what data it uses, what it produced, and the assumptions behind it. For exact commands and file layout, see `README.md`.

---

## The big picture

Four independent scripts, each tackling a different cybersecurity detection problem. In every section, the same pattern is followed:

1. Load a public, real-world security dataset.
2. Clean and prepare the data.
3. Train **two different models** — one "classic" ML model and one deep-learning model — on the same problem.
4. Compare how well each model does, and save charts + metrics.

This lets the report discuss *when a simpler model is good enough* versus *when deep learning actually helps*.

---

## Section 1 — Phishing / Spam Email Detection

**Question:** Given the text of an email, is it spam/phishing or legitimate ("ham")?

**Data:** The Enron Spam Dataset — ~33,700 real emails, labelled spam or ham.

**Models compared:**
- *Classic:* Turn email text into TF-IDF features (word importance scores) + Logistic Regression.
- *Deep learning:* Turn words into embeddings (dense numeric representations) + an LSTM network that reads the email sequentially, like a human reading word by word.

**What happened when we ran it:**
- TF-IDF + Logistic Regression: **99.0% accuracy**
- Embedding + LSTM: **98.5% accuracy**

**Takeaway:** Both models do very well here — spam detection from text is a problem simpler models already solve nearly perfectly, so the LSTM's extra complexity doesn't buy much.

---

## Section 2 — Network Intrusion Detection

**Question:** Given a network connection's traffic statistics, is it Normal, or one of four attack types — DoS, Probe, R2L (remote-to-local break-in), or U2R (user-to-root privilege escalation)?

**Data:** NSL-KDD — a well-known, cleaned-up version of the classic KDD'99 intrusion dataset. Crucially, it's tested on the **official test split**, which deliberately includes attack patterns *never seen during training* — this is the standard, harder way to evaluate on this dataset.

**Models compared:**
- *Classic:* Random Forest (an ensemble of decision trees).
- *Deep learning:* 1-D Convolutional Neural Network (CNN) — treats the connection's features like a short signal and scans it for patterns.

**What happened when we ran it:**
- Random Forest: **76.0% accuracy**
- 1-D CNN: **75.4% accuracy**
- Both models do well on Normal/DoS traffic but struggle badly on rare attack types (R2L, U2R) — this is expected and well-documented behaviour for this benchmark, not a bug.

**Takeaway:** The ~75-80% accuracy ceiling here is a known property of the dataset (rare, unseen attack types make it deliberately hard), not a modelling failure.

---

## Section 3 — Anomaly Detection (Zero-Day Style)

**Question:** Without ever telling the model what an "attack" looks like, can it learn what *normal* network traffic looks like well enough to flag anything unusual?

**Data:** UNSW-NB15 — a modern network traffic dataset with realistic contemporary attack types.

**Key design choice:** Both models are trained **only on normal traffic**. Attack labels are only used afterwards, to check how well the models spotted the abnormal traffic. This mimics a real "zero-day" scenario — you don't have labelled examples of the new attack, only a sense of what's normal.

**Models compared:**
- *Classic:* Isolation Forest — isolates unusual data points by how easily they can be "split off" from the rest.
- *Deep learning:* Autoencoder — a neural net that learns to compress and reconstruct normal traffic; if something doesn't reconstruct well, it's flagged as anomalous.

**What happened when we ran it:**
- Isolation Forest: 67.4% accuracy, caught ~50% of real attacks (TPR)
- Autoencoder: **81.5% accuracy**, caught **73% of real attacks (TPR)**, with a stronger ROC-AUC (0.90 vs 0.82)

**Takeaway:** The Autoencoder clearly outperforms Isolation Forest here — deep learning earns its keep on this harder, fully unsupervised problem.

---

## Section 4 — Ransomware Detection

**Question:** Based on low-level system activity (Sysmon events — file changes, registry edits, process creation, etc.), can we tell ransomware behaviour apart from normal software behaviour?

**Data:** CSU Ransomware Behavioural Dataset — ~350,000 real system events captured in a sandbox, covering several modern ransomware families (LockBit, Medusa, Akira, BlackBasta).

**Models compared:**
- *Classic:* Gradient Boosting, judging each system event on its own.
- *Deep learning:* LSTM, looking at a **sliding window of 10 consecutive events** — i.e., judging behaviour as a short sequence over time, not just one event in isolation.

**What happened when we ran it:**
- Gradient Boosting (per event): 95.9% accuracy
- LSTM (event windows): **99.9% accuracy**, near-perfect precision/recall on ransomware

**Takeaway:** Looking at *sequences* of behaviour rather than single events makes a big difference — ransomware has a distinctive pattern over time (e.g., rapid file changes) that's much clearer in a window than in any single event.

---

## Final assumptions (apply across all sections)

- **All datasets are public, non-sensitive research datasets** — no real client, personal, or proprietary data is used anywhere in this project.
- **Fixed random seed (42)** is used everywhere. The numbers above are the same run reported in `Report.docx` and saved in `outputs/`. Re-running reproduces the classic ML models exactly on the same library versions; neural network results still vary slightly run-to-run, which is normal.
- **Section 1** trains on a balanced sample of 16,000 emails (not the full 33,700) to keep LSTM training practical without a GPU. This can be changed by setting `SAMPLE_SIZE = None` in the script.
- **Section 2**'s accuracy ceiling (~75-80%) is expected: the official test set intentionally contains attack types the model never saw in training.
- **Section 3** strictly trains only on normal traffic for both models — attack labels are used exclusively to *evaluate*, never to train. The anomaly cutoff is the 95th percentile of reconstruction error on normal data.
- **Section 4** removes duplicate events before splitting into train/test, and uses a time-ordered (chronological) 80/20 split per class so overlapping sliding windows can't leak between train and test.
- **No GPU required** — everything trains on CPU in a few minutes per section.
- Every run's charts and metrics are saved automatically into `outputs/sectionN/`, alongside a `run_log.txt` capturing the exact console output for that run.
