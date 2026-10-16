"""Deep URL-only baselines, each trained alone (no XGBoost stacking, no host data).

  char_cnn     character CNN: embedding -> convolutions (kernel 3/4/5) -> global max pool
  bilstm       character BiLSTM: embedding -> BiLSTM -> max + mean pool
  urlnet       URLNet, reduced: character CNN branch + word CNN branch (words hashed into
               WORD_BUCKETS buckets), concatenated -> dense layers (no char-in-word branch)
  transformer  small Transformer encoder over characters (2 layers, d=64, 4 heads)
Same character encoding as Layer C (url_classifier.model.encode_urls, scheme and www. removed).
Training: AdamW, class-weighted BCE, early stopping on val ROC-AUC (patience 2), one seed.
Run:    python src/baselines/deep.py --model char_cnn [--epochs 8]
Output: reports/baselines/<model>.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import b_flags_for, evaluate_scores, load_splits, save  # noqa: E402
from url_classifier.model import DL_ARCHS, encode_urls, encode_words  # noqa: E402

MODELS = ("char_cnn", "bilstm", "urlnet", "transformer")  # architectures: url_classifier/model.py


def predict(model: nn.Module, c: np.ndarray, w: np.ndarray, device: str, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(c), bs):
            cb = torch.from_numpy(c[i:i + bs].astype(np.int64)).to(device)
            wb = torch.from_numpy(w[i:i + bs].astype(np.int64)).to(device)
            out.append(torch.sigmoid(model(cb, wb)).float().cpu().numpy())
    return np.concatenate(out)


def train(name: str, data: dict, y_tr: np.ndarray, y_va: np.ndarray, device: str, epochs: int,
          seed: int = 42, bs: int = 512) -> tuple[nn.Module, list]:
    torch.manual_seed(seed)
    model = DL_ARCHS[name]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3 if name != "transformer" else 5e-4,
                            weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y_tr == 0).sum() / max(1, y_tr.sum())],
                                                           device=device))
    (c_tr, w_tr), (c_va, w_va) = data["train"], data["val"]
    rng = np.random.default_rng(seed)
    best, best_state, bad, log = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for idx in np.array_split(rng.permutation(len(y_tr)), max(1, len(y_tr) // bs)):
            cb = torch.from_numpy(c_tr[idx].astype(np.int64)).to(device)
            wb = torch.from_numpy(w_tr[idx].astype(np.int64)).to(device)
            yb = torch.from_numpy(y_tr[idx].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(cb, wb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = roc_auc_score(y_va, predict(model, c_va, w_va, device))
        log.append({"epoch": ep, "val_auc": round(float(auc), 5), "seconds": round(time.time() - t0)})
        print(f"  {name} epoch {ep}: val_auc={auc:.5f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    return model, log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_splits()
    data = {s: (encode_urls(d.url.tolist()).astype(np.int16),
                encode_words(d.url.tolist()) if args.model == "urlnet"
                else np.zeros((len(d), 1), np.int32))
            for s, d in splits.items()}
    t0 = time.time()
    model, log = train(args.model, data, splits["train"].y.to_numpy(), splits["val"].y.to_numpy(),
                       device, args.epochs)
    scores = {s: predict(model, *data[s], device) for s in data if s != "train"}
    extra = {"train_seconds": round(time.time() - t0, 1), "device": device, "epochs": log,
             "parameters": sum(p.numel() for p in model.parameters())}
    save(evaluate_scores(args.model, scores, splits, b_flags_for(splits), extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
