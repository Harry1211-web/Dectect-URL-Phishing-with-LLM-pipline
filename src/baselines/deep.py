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
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import b_flags_for, evaluate_scores, load_splits, save  # noqa: E402
from url_classifier.model import MAX_LEN, VOCAB_SIZE, encode_urls  # noqa: E402
from url_utils import strip_scheme_www  # noqa: E402

MODELS = ("char_cnn", "bilstm", "urlnet", "transformer")
WORD_BUCKETS = 100_000
MAX_WORDS = 40
_RE_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def encode_words(urls: list[str]) -> np.ndarray:
    """Word-id matrix (n, MAX_WORDS): URL split on non-alphanumerics, words hashed (0 = padding)."""
    out = np.zeros((len(urls), MAX_WORDS), dtype=np.int32)
    for i, u in enumerate(urls):
        words = [w for w in _RE_WORD_SPLIT.split(strip_scheme_www(u).lower()) if w][:MAX_WORDS]
        out[i, :len(words)] = [zlib.crc32(w.encode()) % (WORD_BUCKETS - 1) + 1 for w in words]
    return out


class CharCnn(nn.Module):
    def __init__(self, dim: int = 32, ch: int = 128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(dim, ch, k, padding=k // 2) for k in (3, 4, 5))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * ch, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, c, w=None):
        h = self.emb(c).transpose(1, 2)
        return self.head(torch.cat([torch.relu(cv(h)).max(2).values for cv in self.convs], 1)).squeeze(-1)


class BiLstm(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, 1))

    def forward(self, c, w=None):
        mask = (c != 0).unsqueeze(-1).float()
        h, _ = self.lstm(self.emb(c))
        h_max = (h - (1 - mask) * 1e4).max(1).values
        h_mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(torch.cat([h_max, h_mean], 1)).squeeze(-1)


class UrlNet(nn.Module):
    def __init__(self, dim: int = 32, ch: int = 128):
        super().__init__()
        self.cemb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=0)
        self.wemb = nn.Embedding(WORD_BUCKETS, dim, padding_idx=0)
        self.cconv = nn.ModuleList(nn.Conv1d(dim, ch, k, padding=k // 2) for k in (3, 4, 5, 6))
        self.wconv = nn.ModuleList(nn.Conv1d(dim, ch, k, padding=k // 2) for k in (1, 2, 3))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(7 * ch, 256), nn.ReLU(),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, c, w):
        hc, hw = self.cemb(c).transpose(1, 2), self.wemb(w).transpose(1, 2)
        pc = [torch.relu(cv(hc)).max(2).values for cv in self.cconv]
        pw = [torch.relu(cv(hw)).max(2).values for cv in self.wconv]
        return self.head(torch.cat(pc + pw, 1)).squeeze(-1)


class TinyTransformer(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, c, w=None):
        pad = c == 0
        pad[:, 0] = False  # an empty URL must still attend to something
        h = self.enc(self.emb(c) + self.pos[:, :c.shape[1]], src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1)).squeeze(-1)


BUILDERS = {"char_cnn": CharCnn, "bilstm": BiLstm, "urlnet": UrlNet, "transformer": TinyTransformer}


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
    model = BUILDERS[name]().to(device)
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
