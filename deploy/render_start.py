"""Render start command: make sure the data folder is on the persistent disk, then run the app.

On the first start (or when DATA_BUNDLE changes) the bundle is downloaded from the bucket
and unpacked onto the disk. Files the app wrote itself (sketches, uploads) stay.
"""
import os
import shutil
import sys
import tarfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "deploy"))
import s3client  # noqa: E402

DISK = Path(os.environ.get("RENDER_DISK", "/var/data"))
DATA = DISK / "app-data"


def ensure_data():
    key = os.environ.get("DATA_BUNDLE", "").strip()
    if not key:
        raise SystemExit("DATA_BUNDLE is not set (run deploy/make_bundle.py on the Mac)")
    marker = DATA / ".bundle"
    if marker.exists() and marker.read_text().strip() == key:
        print(f"data: {key} already on disk", flush=True)
        return
    DATA.mkdir(parents=True, exist_ok=True)
    part = DISK / "bundle.tar.part"
    print(f"data: downloading {key} ...", flush=True)
    s3client.client().download_file(s3client.bucket(), key, str(part))
    with tarfile.open(part) as tar:
        tar.extractall(DATA, filter="data")
    part.unlink()
    marker.write_text(key)
    print("data: ready", flush=True)


def link_data():
    target = APP / "data"
    if target.is_symlink():
        return
    if target.exists():
        shutil.rmtree(target)
    target.symlink_to(DATA)


if __name__ == "__main__":
    ensure_data()
    link_data()
    os.environ.setdefault("HF_HOME", str(DISK / "hf"))
    port = os.environ.get("PORT", "10000")
    os.execvp("uvicorn", ["uvicorn", "jewelsearch.server:app", "--host", "0.0.0.0", "--port", port,
                          "--proxy-headers", "--forwarded-allow-ips", "*"])
