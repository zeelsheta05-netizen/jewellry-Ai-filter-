"""Pack the app's data folder (index, crops, try-on models, ...) and upload it to the
bucket as deploy/data-<date>.tar, for the Render server to fetch on start.

    .venv/bin/python deploy/make_bundle.py            # prints the DATA_BUNDLE value for Render

Left out: logs, body photos (never leave this Mac), the session secret, storage cache.
The manifest is copied with SQLite's backup API, so a running move can't tear it.
"""
import sqlite3
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
import s3client  # noqa: E402

DATA = ROOT / "data"
INCLUDE = ["catalog.jsonl", "catalog_summary.json", "crops", "index", "piece_finder", "brand_designs", "sketch",
           "cad", "tryon/models", "tryon/photos", "tryon/designs.json", "tryon/fidelity.json",
           "web_products/img", "web_products/products.jsonl", "web_products/dino.npy",
           "web_products/siglip.npy", "web_products/title.npy"]


def main():
    s3client.load_env(ROOT)
    s3, bucket = s3client.client(), s3client.bucket()
    key = f"deploy/data-{time.strftime('%Y-%m-%d-%H%M')}.tar"
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "manifest.sqlite"
        src = sqlite3.connect(f"file:{DATA / 'storage' / 'manifest.sqlite'}?mode=ro", uri=True)
        dst = sqlite3.connect(snap)
        src.backup(dst)
        dst.close(), src.close()
        tar_path = Path(tmp) / "data.tar"
        with tarfile.open(tar_path, "w") as tar:
            for rel in INCLUDE:
                p = DATA / rel
                if p.exists():
                    tar.add(p, arcname=rel, filter=lambda t: None if t.name.endswith((".log", ".out")) else t)
                else:
                    print(f"  (missing, skipped) {rel}")
            tar.add(snap, arcname="storage/manifest.sqlite")
        size = tar_path.stat().st_size
        print(f"bundle {size / 1e9:.2f} GB -> s3://{bucket}/{key}", flush=True)
        s3.upload_file(str(tar_path), bucket, key)
    head = s3.head_object(Bucket=bucket, Key=key)
    assert head["ContentLength"] == size, "upload size mismatch"
    print(f"\nUploaded. In Render set:  DATA_BUNDLE={key}")


if __name__ == "__main__":
    main()
