"""The bucket client the deploy scripts use: same settings as jewelsearch/storage.py
(S3_ENDPOINT / S3_REGION / AWS_* from the environment or .env)."""
import os
from pathlib import Path

import boto3
from botocore.config import Config


def load_env(root: Path):
    f = root / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def client():
    endpoint = os.environ.get("S3_ENDPOINT") or None
    compat = {"request_checksum_calculation": "when_required",
              "response_checksum_validation": "when_required"} if endpoint else {}
    cfg = Config(signature_version="s3v4", retries={"max_attempts": 8, "mode": "adaptive"}, **compat)
    return boto3.client("s3", region_name=os.environ.get("S3_REGION") or None, endpoint_url=endpoint, config=cfg)


def bucket() -> str:
    b = os.environ.get("S3_BUCKET", "").strip()
    if not b:
        raise SystemExit("S3_BUCKET is not set")
    return b
