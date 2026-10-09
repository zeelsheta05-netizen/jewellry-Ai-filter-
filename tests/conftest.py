"""Test-wide settings: the 4.5 GB design-detail reader (jewelsearch/details.py) isn't
loaded by engines made in tests; tests that need it pass a stand-in or load it themselves."""
import os

os.environ.setdefault("JEWEL_DETAILS", "0")

import pytest

# The real bucket's settings and keys from .env never reach a test: tests that need
# S3 set their own and use a simulated bucket (moto). Without this, a test could read
# the live S3_ENDPOINT / keys and talk to the real Zata bucket.
_REAL_S3 = ("S3_ENDPOINT", "S3_BUCKET", "S3_REGION", "S3_CONDITIONAL_WRITES", "S3_ORIGINALS_CLASS",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION",
            "GEMINI_API_KEY", "POLLINATIONS_KEY")   # no test may spend real AI credit


@pytest.fixture(autouse=True)
def _no_real_bucket(monkeypatch):
    for k in _REAL_S3:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LOCAL_IMAGEGEN", "0")   # tests never start the real on-Mac image generator


@pytest.fixture(autouse=True)
def _private_drawing_lock(monkeypatch, tmp_path):
    """Tests take their own Mac-wide drawing lock (never the live app's) and don't look for
    real FLUX runs on this Mac (tests replace subprocess.run with stand-ins)."""
    from jewelsearch import sketch
    monkeypatch.setattr(sketch, "DRAW_LOCK_FILE", tmp_path / "drawing.lock")
    monkeypatch.setattr(sketch, "_other_drawing", lambda: False)
