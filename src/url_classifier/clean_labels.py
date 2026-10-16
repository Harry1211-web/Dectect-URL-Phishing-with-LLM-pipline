"""Find likely mislabelled training URLs with out-of-fold predictions (confident learning).

Aggregated feeds (Phishing.Database, PhiUSIIL) contain normal pages labelled "phishing"
(book pages, shop products, blog posts of once-compromised sites). Such points drag the
high-recall threshold down until almost every clean URL is sent to the LLM.

Method: split the TRAIN sites in two halves, train the CNN/BiLSTM on one half and score the
other (and vice versa), so every train URL gets a score from a model that never saw its site.
Rows from the unverified sources whose out-of-fold score strongly contradicts their label are
flagged; train.py drops them. Val/test are never filtered (evaluation stays honest).

Run:    python src/url_classifier/clean_labels.py
Output: dataset/urls/noisy_labels.csv (url, label, source, oof_score)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_classifier.model import encode_inputs  # noqa: E402
from url_classifier.train import DATA_PATH, predict_logits, train_dl  # noqa: E402

OUT_PATH = DATA_PATH.parent / "noisy_labels.csv"
# Human-verified feeds are trusted as-is; only aggregated/academic sources are filtered
UNVERIFIED_SOURCES = {"phishing_database", "phiusiil", "tranco_crawl"}
MAX_SCORE_FOR_PHISHING = 0.02   # a "phishing" row the unseen model scores below this is suspect
MIN_SCORE_FOR_CLEAN = 0.98      # a "clean" row scored above this is suspect


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(DATA_PATH)
    tr = df[df.split == "train"].copy()
    tr["y"] = (tr.label == "phishing").astype(np.int64)
    half = tr.group.map(lambda g: int(hashlib.md5(f"oof:{g}".encode()).hexdigest(), 16) % 2)
    va = df[df.split == "val"]
    x_va, y_va = encode_inputs("cnn_bilstm", va.url.tolist()), (va.label == "phishing").to_numpy().astype(np.int64)

    tr["oof_score"] = np.nan
    for k in (0, 1):
        fit, held = tr[half != k], tr[half == k]
        print(f"fold {k}: train {len(fit)}, score {len(held)}", flush=True)
        model, _ = train_dl("cnn_bilstm", encode_inputs("cnn_bilstm", fit.url.tolist()),
                            fit.y.to_numpy(), x_va, y_va, device, epochs=5, seed=k)
        logits = predict_logits(model, encode_inputs("cnn_bilstm", held.url.tolist()), device)
        tr.loc[held.index, "oof_score"] = 1 / (1 + np.exp(-logits))

    unverified = tr.source.isin(UNVERIFIED_SOURCES)
    noisy = tr[unverified & (((tr.y == 1) & (tr.oof_score < MAX_SCORE_FOR_PHISHING))
                             | ((tr.y == 0) & (tr.oof_score > MIN_SCORE_FOR_CLEAN)))]
    noisy[["url", "label", "source", "oof_score"]].to_csv(OUT_PATH, index=False)
    print(noisy.groupby(["source", "label"]).size().to_string())
    print(f"flagged {len(noisy)} of {len(tr)} train rows -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
