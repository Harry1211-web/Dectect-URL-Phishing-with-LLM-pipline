"""The Vietnamese evaluation set stays isolated from training (zero domain overlap)."""

from pathlib import Path

import pandas as pd
import pytest

from data_collection.vi_testset import OUT_DIR, as_url, training_domains
from url_utils import registrable_domain

SRC = Path(__file__).resolve().parents[1] / "src"


def test_as_url_matches_training_shape():
    assert as_url("evil.top") == "http://evil.top/"
    assert as_url("evil.top/login") == "http://evil.top/login"


@pytest.mark.parametrize("name", ["phishing", "clean"])
def test_no_domain_shared_with_training(name):
    f = OUT_DIR / f"{name}.csv"
    if not f.exists():
        pytest.skip(f"{f} not built yet")
    domains = set(pd.read_csv(f).url.map(registrable_domain))
    assert not domains & training_domains()


def test_training_code_never_reads_the_vi_set():
    for script in ("data_collection/build_dataset.py", "url_classifier/train.py"):
        text = (SRC / script).read_text(encoding="utf-8")
        assert "vi_testset" not in text and "test_sets" not in text and '"vi"' not in text, script
