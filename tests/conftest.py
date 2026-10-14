"""Make `src/` importable and expose shared paths to the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "llm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
