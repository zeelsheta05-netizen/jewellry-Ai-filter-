import hashlib
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path):
    """Read KEY=VALUE lines from .env (secrets live there, never in code)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env(ROOT / ".env")

# Where the dataset lives, as a URL, set in .env only (see jewelsearch/storage.py):
#   JEWEL_STORAGE=s3://<bucket>                 the S3 bucket
#   JEWEL_STORAGE=file:///<folder>              a folder (the SSD before the move)
#   JEWEL_STORAGE_FALLBACK=file:///<folder>     while the move runs: files not yet in the
#                                               bucket are read from here
# Code never opens dataset files by path; it asks jewelsearch.storage.
STORAGE_URL = os.environ.get("JEWEL_STORAGE", "").strip()
STORAGE_FALLBACK_URL = os.environ.get("JEWEL_STORAGE_FALLBACK", "").strip()
DATA = ROOT / "data"
CROPS = DATA / "crops"   # tight square crops around the piece: what the image models read (pages show the originals)
INDEX = DATA / "index"
CATALOG = DATA / "catalog.jsonl"

MODEL_NAME = "ViT-B-16-SigLIP2-256"
MODEL_PRETRAINED = "webli"

CATEGORIES = ["ring", "earrings", "pendant", "necklace", "bracelet"]
METALS = ["rose_gold", "white_gold", "yellow_gold"]
# Metal used for the embedding when a design has several; metal is a filter,
# so every design is embedded in the same colour to keep shapes comparable.
EMBED_METAL_ORDER = ["white_gold", "yellow_gold", "rose_gold"]


def media_token(relpath: str) -> str:
    """Opaque, stable ID of a dataset file for links: pages never see dataset paths."""
    return hashlib.sha1(relpath.encode()).hexdigest()[:16]


def thumb_name(relpath: str) -> str:
    return media_token(relpath) + ".webp"


# Every model on the GPU runs under this one lock. PyTorch's Apple GPU (MPS) backend
# has one command queue per process and is not thread-safe: two requests running
# models at the same moment (SigLIP for one, DINOv2 or the judge for another)
# crashed the app with "failed assertion _status < MTLCommandBufferStatusCommitted".
GPU = threading.RLock()


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
