"""Deep URL models of Layer C (the DL part; its phishing score feeds the ML classifier).

  cnn_bilstm   char ids -> embedding -> parallel 1D convolutions (kernel 3/5/7) -> BiLSTM ->
               max + mean pooling -> dense EMBED_DIM vector -> linear head
  char_cnn     embedding -> convolutions (kernel 3/4/5) -> global max pool
  bilstm       embedding -> BiLSTM -> max + mean pool
  urlnet       URLNet, reduced: character CNN branch + word CNN branch (words hashed into
               WORD_BUCKETS buckets), concatenated -> dense layers (no char-in-word branch)
  transformer  small Transformer encoder over characters (2 layers, d=64, 4 heads)
Every model takes (char ids, word ids) and returns one logit; only urlnet reads the word ids.
URL text: scheme and 'www.' removed.
"""

from __future__ import annotations

import re
import string
import sys
import zlib
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

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)                       # logits


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


DL_ARCHS = {"cnn_bilstm": CharCnnLstm, "char_cnn": CharCnn, "bilstm": BiLstm, "urlnet": UrlNet,
            "transformer": TinyTransformer}


def uses_words(arch: str) -> bool:
    return arch == "urlnet"


def encode_inputs(arch: str, urls: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(char ids, word ids) for `arch`; a 1-column dummy word matrix when the model has no word branch."""
    return (encode_urls(urls),
            encode_words(urls) if uses_words(arch) else np.zeros((len(urls), 1), np.int64))
