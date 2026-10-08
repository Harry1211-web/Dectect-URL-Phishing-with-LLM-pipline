"""Character-level CNN + BiLSTM feature extractor for Layer C.

URL text (scheme and 'www.' removed) -> char ids -> embedding -> parallel 1D convolutions
(n-gram detectors, kernel 3/5/7) -> BiLSTM (order of the n-grams) -> max+mean pooling ->
dense EMBED_DIM vector. A small linear head is used only to train the extractor; the EMBED_DIM
vector is then fed to XGBoost together with the hand-crafted features.
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_utils import strip_scheme_www  # noqa: E402

MAX_LEN = 200
EMBED_DIM = 64
_ALPHABET = string.ascii_letters + string.digits + string.punctuation
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(_ALPHABET)}  # 0 = padding, 1 = unknown char
VOCAB_SIZE = len(CHAR_TO_ID) + 2


def encode_urls(urls: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    """Char-id matrix (n, max_len). Long URLs keep their head and tail, where most signal is."""
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    half = max_len // 2
    for i, u in enumerate(urls):
        s = strip_scheme_www(u)
        if len(s) > max_len:
            s = s[:half] + s[-(max_len - half):]
        out[i, :len(s)] = [CHAR_TO_ID.get(c, 1) for c in s]
    return out


class CharCnnLstm(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, char_dim: int = 32, conv_ch: int = 64,
                 lstm_hidden: int = 64, embed_dim: int = EMBED_DIM, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(char_dim, conv_ch, k, padding=k // 2) for k in (3, 5, 7))
        self.lstm = nn.LSTM(conv_ch * 3, lstm_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden * 4, embed_dim),
                                  nn.ReLU())
        self.head = nn.Linear(embed_dim, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, MAX_LEN) char ids -> (batch, EMBED_DIM) URL embedding."""
        mask = (x != 0).unsqueeze(-1).float()
        h = self.emb(x).transpose(1, 2)                                   # (b, char_dim, L)
        h = torch.cat([torch.relu(c(h)) for c in self.convs], 1).transpose(1, 2).contiguous()
        h, _ = self.lstm(h)                                               # (b, L, 2*hidden)
        h_max = (h - (1 - mask) * 1e4).max(1).values                      # ignore padding
        h_mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(torch.cat([h_max, h_mean], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)                       # logits
