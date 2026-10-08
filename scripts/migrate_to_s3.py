#!/usr/bin/env python3
"""Move the dataset from the SSD into the S3 bucket: compressed, de-duplicated,
checked, and listed in the storage index (jewelsearch/storage.py).

The short way, once the S3 settings are in .env (S3_BUCKET, S3_REGION, S3_ENDPOINT
for non-AWS providers, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):

    migrate_to_s3.py connect      ~15 min: checks, bucket setup, switches the app to the bucket
                                  with the SSD as fallback (restart the app afterwards)
    migrate_to_s3.py move         hours to days, in the background: copies everything, verifies
                  --max-gb N      only part of it (e.g. on a free trial): what the app shows first,
                                  stops at N GB; run again with a bigger N or none for the rest
                  --category ring only that category's pictures and 3D videos (with --max-gb:
                                  its pictures first, then videos up to the limit)
    migrate_to_s3.py finish       after move: removes the SSD fallback; the app uses only the bucket
    migrate_to_s3.py restore      on a new or rebuilt server: brings back the indexes from the bucket

The single steps, in order (each can be stopped and run again: finished work is skipped):

    migrate_to_s3.py check                       settings, bucket reachable, test write/read/delete
    migrate_to_s3.py setup [--versioning]        block public access, clean up failed uploads after 3 days,
                                                 optionally keep old versions (undo for deletes)
    migrate_to_s3.py scan  --source /Volumes/Storage [--estimate]
                                                 list the SSD (sizes only) and estimate the savings
    migrate_to_s3.py web   --source /Volumes/Storage [--videos original|encode] [--limit N]
                                                 what the app shows: a WebP copy of every catalogue
                                                 render, and the 3D videos. After this the app can
                                                 run on S3 (set JEWEL_STORAGE=s3://<bucket>)
    migrate_to_s3.py originals --source /Volumes/Storage [--limit N]
                                                 every file from the SSD once, unchanged; identical
                                                 copies stored once; CAD/text zstd-compressed
    migrate_to_s3.py relayout [--workers 8]      give files already in the bucket their clean names
                                                 (renamed inside the bucket, nothing is sent again;
                                                 run while the app is stopped)
    migrate_to_s3.py verify [--sample 200]       every object is in the bucket with the right size;
                                                 a random sample is downloaded and its SHA-256 checked
    migrate_to_s3.py index                       upload snapshots of the storage index and the
                                                 app's search index (for a new or rebuilt server)
    migrate_to_s3.py status                      what is moved, what is left, what compression saved

Bucket: --bucket, else S3_BUCKET, else JEWEL_STORAGE=s3://<bucket>. Region and
keys come from .env (S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY; S3_ENDPOINT
only for S3-compatible services). The SSD is only read, never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import quote
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jewelsearch import storage as ds  # noqa: E402
from jewelsearch.config import CATALOG, DATA, INDEX  # noqa: E402

ENV = ROOT / ".env"

MiB = 1024 * 1024
ZSTD_LEVEL = 9                 # good ratio at ~100 MB/s on the M2; CAD files shrink 2-4x
ZSTD_MIN_SAVING = 0.10         # keep the compressed copy only when it saves >= 10 %
WEBP_QUALITY = 90              # display copy; the lossless original stays in the bucket too
VIDEO_MAX_WIDTH = 1920
VIDEO_CRF = 22                 # H.264 quality (lower = better and bigger); 22 is visually clean
VIDEO_MIN_SAVING = 0.20        # keep the re-encoded video only when it is >= 20 % smaller
WEB_CACHE_CONTROL = "private, max-age=86400"


def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


# ---------------------------------------------------------------- settings

def settings(args) -> dict:
    bucket = args.bucket or os.environ.get("S3_BUCKET") or ""
    url = os.environ.get("JEWEL_STORAGE", "")
    if not bucket and url.startswith("s3://"):
        bucket = url[5:].split("/")[0]
    if not bucket:
        raise SystemExit("No bucket: pass --bucket, or set S3_BUCKET=<name> in .env")
    return {"bucket": bucket, "region": os.environ.get("S3_REGION") or os.environ.get("AWS_REGION"),
            "endpoint": os.environ.get("S3_ENDPOINT") or None}


def open_s3(cfg: dict) -> ds.S3Storage:
    return ds.S3Storage(cfg["bucket"], cfg["region"], cfg["endpoint"])


def default_source() -> str:
    """The SSD, as configured: the fallback during the move, or the current storage before it."""
    for key in ("JEWEL_STORAGE_FALLBACK", "JEWEL_STORAGE"):
        url = os.environ.get(key, "")
        if url.startswith("file://"):
            return ds.from_url(url).root.as_posix()
    return ""


def source_of(args) -> ds.LocalStorage:
    src = getattr(args, "source", "") or default_source()
    if not src:
        raise SystemExit("No SSD folder: pass --source, or keep JEWEL_STORAGE(_FALLBACK)=file:///<folder> in .env")
    st = ds.LocalStorage(src)
    if not st.ready():
        raise SystemExit(f"{src} is not mounted: plug in the SSD")
    return st


def set_env(updates: dict):
    """Change only these keys in .env (None removes one); everything else is kept. Mode 600."""
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    out, seen = [], set()
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            seen.add(key)
            if updates[key] is not None:
                out.append(f"{key}={updates[key]}")
            continue
        out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in seen and v is not None]
    tmp = ENV.with_name(".env.tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(ENV)
    for k, v in updates.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# worker processes make their own client (boto3 clients don't cross processes)
_worker_s3: ds.S3Storage | None = None


def _s3_in_worker(cfg: dict) -> ds.S3Storage:
    global _worker_s3
    if _worker_s3 is None:
        _worker_s3 = open_s3(cfg)
    return _worker_s3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * MiB):
            h.update(chunk)
    return h.hexdigest()


class HashingReader(io.RawIOBase):
    """Reads a file once while hashing it, for a streamed upload. Not seekable,
    so the uploader reads it front to back exactly once."""

    def __init__(self, f):
        self.f, self.sha, self.n = f, hashlib.sha256(), 0

    def readable(self):
        return True

    def read(self, size=-1):
        b = self.f.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)


def transfer_config():
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(multipart_threshold=64 * MiB, multipart_chunksize=64 * MiB, max_concurrency=4)


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ---------------------------------------------------------------- partial moves (--max-gb)

BUCKET_FULL = ("The provider refused more data (QuotaExceeded): the storage limit of the plan or the "
               "free trial is reached. Everything sent so far is recorded. Buy a bigger plan or lower "
               "--max-gb, then run the same command again: it continues where it stopped.")


def bucket_full(e: Exception) -> bool:
    """The provider said no more space (Ceph/Zata: QuotaExceeded). Trying the
    remaining files would only fail the same way, so the step stops."""
    return "QuotaExceeded" in f"{type(e).__name__} {e}"


def stop_when_full(fn, stop: threading.Event):
    """Wrap an upload job: once the provider has refused one file for lack of space,
    the files after it are not even tried (each would fail the same way)."""
    def run(*a):
        if stop.is_set():
            raise RuntimeError("QuotaExceeded earlier in this run: not tried")
        try:
            return fn(*a)
        except Exception as e:
            if bucket_full(e):
                stop.set()
            raise
    return run


def require_bucket(s3: ds.S3Storage):
    """Stop before sending anything when the bucket can't be reached with the
    settings in .env (wrong S3_BUCKET, old keys...): otherwise every file fails
    one by one."""
    try:
        s3.client.head_bucket(Bucket=s3.bucket)
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "") or type(e).__name__
        hint = {"404": "no bucket with this name: check S3_BUCKET in .env",
                "NoSuchBucket": "no bucket with this name: check S3_BUCKET in .env",
                "403": "the keys can't open it: check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env"}.get(code, "")
        raise SystemExit(f"Can't reach the bucket '{s3.bucket}' ({code}){': ' + hint if hint else ''}. "
                         "Nothing was sent. Fix .env, run `migrate_to_s3.py check`, then start again.")


def room_left(args, man: ds.Manifest) -> int | None:
    """Bytes the bucket may still take under --max-gb (None: no limit). Counted from
    what the storage index says is stored, so a second run continues where the
    first stopped. GB = 10^9 bytes, a little under a GiB, so the limit errs low."""
    limit = getattr(args, "max_gb", 0) or 0
    if limit <= 0:
        return None
    used = man.db.execute("SELECT COALESCE(SUM(stored), 0) FROM objects").fetchone()[0]
    return max(0, int(limit * 1e9) - used)


def within(items: list, room: int | None, cost) -> tuple[list, int]:
    """The items, in their order, that fit in `room` bytes. One too big is skipped,
    so smaller ones after it can still go. Returns (fitting items, number skipped)."""
    if room is None:
        return items, 0
    out, skipped = [], 0
    for it in items:
        c = cost(it)
        if c <= room:
            out.append(it)
            room -= c
        else:
            skipped += 1
    return out, skipped


def category_paths(args, rows: list[tuple]) -> set[str] | None:
    """--category ring: only that category's catalogue pictures and 3D videos (what
    the app shows). None: no category limit. Other files (CAD, job cards) wait."""
    cat = (getattr(args, "category", "") or "").strip().lower()
    if not cat:
        return None
    known = sorted({r[3] for r in rows})
    if cat not in known:
        raise SystemExit(f"--category {cat}: not a catalogue category. Choose one of: {', '.join(known)}")
    return {r[6] for r in rows if r[3] == cat}


def move_priority(catalogue: dict[str, int]):
    """Sort key: what the app shows goes first, so a partial move helps the most.
    Catalogue pictures, other pictures, the catalogue's 3D videos, other videos,
    then the rest (CAD files, job cards, documents)."""
    def key(path: str):
        view, kind = catalogue.get(path), ds.kind_of(path)
        if view is not None and view != 5:
            rank = 0
        elif kind == "image":
            rank = 1
        elif view == 5:
            rank = 2
        elif kind == "video":
            rank = 3
        else:
            rank = 4
        return rank, path
    return key


# ---------------------------------------------------------------- check / setup

def cmd_check(args):
    cfg = settings(args)
    s3 = open_s3(cfg)
    print(f"bucket    {cfg['bucket']}")
    print(f"region    {cfg['region'] or '(default)'}")
    print(f"endpoint  {cfg['endpoint'] or 'AWS S3'}")
    print(f"originals storage class  {s3.originals_class or 'STANDARD'}")
    c = s3.client
    c.head_bucket(Bucket=cfg["bucket"])
    key = ds.INDEX + ".write-test"
    body = os.urandom(16 * MiB)
    t0 = time.time()
    c.put_object(Bucket=cfg["bucket"], Key=key, Body=body)
    up = len(body) / max(time.time() - t0, 1e-3)
    t0 = time.time()
    assert c.get_object(Bucket=cfg["bucket"], Key=key)["Body"].read() == body
    down = len(body) / max(time.time() - t0, 1e-3)
    try:
        public = bucket_is_public(s3, key)
        link_ok = signed_link_works(s3, key, body)
        conditional = conditional_writes_work(c, cfg["bucket"], key, body)
        multipart_ok = multipart_works(c, cfg["bucket"], ds.INDEX + ".multipart-test", body)
    finally:
        remove_every_version(c, cfg["bucket"], key)
        remove_every_version(c, cfg["bucket"], ds.INDEX + ".multipart-test")
    print(f"browser links come from  {s3.media_origin()}")
    print(f"speed (one connection)   upload {fmt(up)}/s, download {fmt(down)}/s")
    print("large files (multipart)  " + ("work" if multipart_ok else "FAILED"))
    print("refuse overwrites (S3_CONDITIONAL_WRITES)  " + (
        "supported: S3_CONDITIONAL_WRITES=1 can be set" if conditional else
        "not supported here: keep it off (versioning keeps any overwritten file)"))
    pending = ds.Manifest().db.execute("SELECT COALESCE(SUM(s.size),0) FROM scan s LEFT JOIN files f "
                                       "ON f.path = s.path WHERE f.path IS NULL").fetchone()[0]
    if pending:
        print(f"left to move             {fmt(pending)}: about {pending / up / 3600:.0f} h at this speed "
              f"(the move uses several connections, usually faster)")
    print(f"ffmpeg (video compression)  {ffmpeg_exe() or 'not found'}")
    if public:
        raise SystemExit("DANGER: the test file could be downloaded WITHOUT a signed link: the bucket is public. "
                         "Make it private in the provider's console, then run check again.")
    print("private: files can only be opened through signed links")
    if not link_ok:
        raise SystemExit("A signed link made by the app did not open the file the way a browser does. "
                         "Full-size images and videos would not show: check the region setting (S3_REGION) "
                         "or ask the provider to allow signed (presigned) GET links.")
    print("signed links: open in a browser-style request, including the partial downloads videos use to seek")
    if not multipart_ok:
        raise SystemExit("Large files (sent in parts) could not be stored and read back: videos and CAD files "
                         "over 64 MB would fail. Ask the provider about S3 multipart uploads.")
    print("OK: the bucket can be written, read and cleaned up with these keys.")


def _https_get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    """A plain https GET, the way a browser fetches an <img>/<video> (no S3 library,
    no keys). Certificates are checked against certifi's list (this Mac's Python
    has none of its own). Connection problems raise: a check that can't connect
    must fail, never pass."""
    import ssl
    import urllib.error
    import urllib.request
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30, context=ctx) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def conditional_writes_work(c, bucket: str, key: str, body: bytes) -> bool:
    """Does the bucket refuse a write to a name that already exists (If-None-Match: *)?
    Some S3-compatible services ignore the header and overwrite; the test file is
    rewritten with the same bytes, so nothing is lost either way."""
    from botocore.exceptions import ClientError
    try:
        c.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
    except ClientError as e:
        return e.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412")
    return False


def multipart_works(c, bucket: str, key: str, body: bytes) -> bool:
    """Send a file in two parts (the way large videos and CAD files go) and read it back."""
    half = len(body) // 2   # 8 MiB each, above the 5 MiB S3 minimum part size
    up = c.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    try:
        parts = [{"PartNumber": n, "ETag": c.upload_part(Bucket=bucket, Key=key, UploadId=up, PartNumber=n,
                                                         Body=chunk)["ETag"]}
                 for n, chunk in ((1, body[:half]), (2, body[half:]))]
        c.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=up, MultipartUpload={"Parts": parts})
    except Exception:
        try:
            c.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=up)
        except Exception:
            pass
        return False
    try:
        return c.get_object(Bucket=bucket, Key=key)["Body"].read() == body
    except Exception:
        return False


def remove_every_version(c, bucket: str, key: str):
    """Delete a test file for good. With versioning on, a plain delete only hides it
    and the old copy keeps using paid space, so each version is removed by its id."""
    try:
        r = c.list_object_versions(Bucket=bucket, Prefix=key)
        found = [v for v in r.get("Versions", []) + r.get("DeleteMarkers", []) if v["Key"] == key]
    except Exception:
        found = []
    for v in found:
        c.delete_object(Bucket=bucket, Key=key, VersionId=v["VersionId"])
    if not found:
        try:
            c.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass


def bucket_is_public(s3: ds.S3Storage, key: str) -> bool:
    """Try to download a file without any signature, the way a stranger would."""
    url = s3.client.generate_presigned_url("get_object", {"Bucket": s3.bucket, "Key": key}, ExpiresIn=60).split("?")[0]
    return _https_get(url)[0] == 200


def signed_link_works(s3: ds.S3Storage, key: str, body: bytes) -> bool:
    """Open a signed link as a page would: the whole file (<img>, "Full resolution"),
    and a byte range (a <video> seeking)."""
    url = s3.client.generate_presigned_url("get_object", {"Bucket": s3.bucket, "Key": key}, ExpiresIn=300)
    status, data = _https_get(url)
    if status != 200 or data != body:
        return False
    status, data = _https_get(url, {"Range": "bytes=100-199"})
    return status == 206 and data == body[100:200]


def cmd_setup(args):
    cfg = settings(args)
    c = open_s3(cfg).client
    b = cfg["bucket"]
    try:
        c.put_public_access_block(Bucket=b, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        print("public access: blocked (files are only reachable through signed links)")
    except Exception as e:
        print(f"public access: could not set ({e}); block it in the provider's console")
    rules = []
    try:
        rules = c.get_bucket_lifecycle_configuration(Bucket=b).get("Rules", [])
    except Exception:
        pass
    rules = [r for r in rules if r.get("ID") != "jewel-abort-incomplete-uploads"]
    rules.append({"ID": "jewel-abort-incomplete-uploads", "Status": "Enabled", "Filter": {"Prefix": ""},
                  "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 3}})
    try:
        c.put_bucket_lifecycle_configuration(Bucket=b, LifecycleConfiguration={"Rules": rules})
        print("lifecycle: unfinished uploads are removed after 3 days")
    except Exception as e:
        print(f"lifecycle: could not set ({e})")
    try:
        enc = c.get_bucket_encryption(Bucket=b)["ServerSideEncryptionConfiguration"]["Rules"][0]
        print(f"encryption at rest: {enc['ApplyServerSideEncryptionByDefault']['SSEAlgorithm']}")
    except Exception:
        print("encryption at rest: not reported (AWS encrypts every new bucket by default)")
    if args.versioning:
        try:   # DigitalOcean Spaces and Akamai only offer this through the API
            c.put_bucket_versioning(Bucket=b, VersioningConfiguration={"Status": "Enabled"})
        except Exception as e:
            print(f"versioning: could not turn on ({e})")
    try:
        v = c.get_bucket_versioning(Bucket=b).get("Status", "Off")
        print(f"versioning: {v}" + ("" if v == "Enabled" else
              " (turn it on in the console to undo accidental deletes; it costs little because nothing is overwritten)"))
    except Exception:
        print("versioning: not offered by this provider. Protect originals/ with a bucket lock instead "
              "(Cloudflare R2: bucket > Settings > Bucket lock rules, prefix originals/, no end date)")


# ---------------------------------------------------------------- scan

def cmd_scan(args):
    src = source_of(args)
    man = ds.Manifest()
    old = {r["path"]: (r["size"], r["mtime"], r["sha256"]) for r in man.db.execute("SELECT * FROM scan")}
    seen, t0 = set(), time.time()
    rows = []
    for rel in src.walk():
        try:
            st = (src.root / rel).stat()
        except OSError:
            continue
        prev = old.get(rel)
        sha = prev[2] if prev and prev[0] == st.st_size and prev[1] == st.st_mtime else None
        rows.append((rel, st.st_size, st.st_mtime, sha))
        seen.add(rel)
        if len(rows) % 5000 == 0:
            print(f"  {len(rows)} files listed", flush=True)
    with man.write() as con:
        con.executemany("INSERT OR REPLACE INTO scan VALUES (?,?,?,?)", rows)
        gone = [(p,) for p in old if p not in seen]
        con.executemany("DELETE FROM scan WHERE path = ?", gone)
    by_kind, sizes = defaultdict(lambda: [0, 0]), Counter()
    for rel, size, _, _ in rows:
        k = by_kind[ds.kind_of(rel)]
        k[0] += 1
        k[1] += size
        sizes[size] += 1
    total = sum(r[1] for r in rows)
    print(f"\n{len(rows)} files, {fmt(total)} on {src.root}  ({time.time() - t0:.0f}s)")
    for kind, (n, b) in sorted(by_kind.items(), key=lambda x: -x[1][1]):
        print(f"  {kind:<9} {n:>7} files  {fmt(b):>10}")
    maybe_dup = sum(n for s, n in sizes.items() if n > 1 and s > 0)
    print(f"  {maybe_dup} files share their size with another file: possible duplicates, checked by SHA-256 on upload")
    if args.estimate:
        estimate(src, rows)


def estimate(src: ds.LocalStorage, rows):
    """Compress a random sample to show what the bucket will hold."""
    from compression import zstd
    rnd = random.Random(1)
    renders = [r for r in rows if r[0].lower().endswith(".png") and "#vi" in r[0].lower()]
    packable = [r for r in rows if ds.compressible(r[0]) and r[1] >= 4096 and r[1] < 300 * MiB]
    print("\nEstimate from a random sample:")
    if renders:
        sample = rnd.sample(renders, min(20, len(renders)))
        before = after = 0
        for rel, size, _, _ in sample:
            before += size
            after += len(webp_bytes((src.root / rel).read_bytes())[0])
        ratio = after / before
        all_png = sum(r[1] for r in renders)
        print(f"  renders  WebP display copy = {ratio:.0%} of the PNG -> {fmt(all_png)} becomes about {fmt(all_png * ratio)}")
    if packable:
        sample = rnd.sample(packable, min(20, len(packable)))
        before = after = 0
        for rel, size, _, _ in sample:
            data = (src.root / rel).read_bytes()
            before += size
            after += min(size, len(zstd.compress(data, level=ZSTD_LEVEL)))
        ratio = after / before
        total = sum(r[1] for r in packable)
        print(f"  CAD/text zstd = {ratio:.0%} of the original -> {fmt(total)} becomes about {fmt(total * ratio)}")
    videos = [r for r in rows if r[0].lower().endswith(".mp4")]
    if videos:
        exe = ffmpeg_exe()
        if not exe:
            print("  videos   ffmpeg not found: can't estimate re-encoded size")
        else:
            sample = rnd.sample(videos, min(2, len(videos)))
            before = after = 0
            for rel, size, _, _ in sample:
                out = encode_video(exe, src.root / rel)
                if out:
                    before += size
                    after += out.stat().st_size
                    out.unlink()
            if before:
                ratio = after / before
                total = sum(r[1] for r in videos)
                print(f"  videos   H.264 web copy = {ratio:.0%} of the original -> {fmt(total)} served as about {fmt(total * ratio)}")


# ---------------------------------------------------------------- web copies

def webp_bytes(data: bytes, quality: int = WEBP_QUALITY) -> tuple[bytes, int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
        out = io.BytesIO()
        im.save(out, "WEBP", quality=quality, method=4)
        return out.getvalue(), im.width, im.height


def encode_video(exe: str, src: Path) -> Path | None:
    ds.TMP.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=ds.TMP)
    os.close(fd)
    cmd = [exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0",
           "-vf", f"scale='trunc(min({VIDEO_MAX_WIDTH},iw)/2)*2':-2", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        Path(tmp).unlink(missing_ok=True)
        print(f"  ffmpeg failed on {src.name}: {r.stderr.strip()[:200]}", flush=True)
        return None
    return Path(tmp)


def load_designs() -> list[dict]:
    """Catalogue designs with the category the app uses (index), else the catalogue's guess."""
    f = INDEX / "meta.jsonl"
    src = f if f.exists() else CATALOG
    out = []
    for line in src.read_text().splitlines():
        m = json.loads(line)
        if m.get("images") or m.get("videos"):
            out.append(m)
    return out


def design_rows(designs: list[dict]) -> list[tuple]:
    rows = []
    for m in designs:
        key = m.get("key") or f"{m['folders'][0]}/{m['design_id']}"
        slug = ds.design_slug(m["design_id"], m["folders"][0])
        cat = m.get("category") or "unknown"
        for metal, views in m.get("images", {}).items():
            for v, path in views.items():
                rows.append((key, m["design_id"], slug, cat, metal, int(v), path))
        for metal, path in m.get("videos", {}).items():
            rows.append((key, m["design_id"], slug, cat, metal, 5, path))
    return rows


def _render_job(cfg, src_root, path, key):
    """In a worker process: read the PNG, make the WebP copy, upload it."""
    s3 = _s3_in_worker(cfg)
    data = (Path(src_root) / path).read_bytes()
    webp, w, h = webp_bytes(data)
    s3.client.put_object(Bucket=cfg["bucket"], Key=key, Body=webp, ContentType="image/webp",
                         CacheControl=WEB_CACHE_CONTROL)
    return {"path": path, "key": key, "sha": hashlib.sha256(data).hexdigest(), "size": len(data),
            "stored": len(webp), "w": w, "h": h}


def design_folders(rows: list[tuple]) -> dict[str, tuple[str, str, str]]:
    """design_key -> (category, readable folder name, the design's folder on the SSD).
    The folder name is the design id; the slug's short hash is added only when the
    same id appears twice in a category (bare ids such as "14" repeat across lots)."""
    first: dict[str, tuple] = {}
    for key, did, slug, cat, metal, view, path in rows:
        first.setdefault(key, (ds.clean_part(cat or "other"), ds.clean_part(did), slug, os.path.dirname(path)))
    seen = Counter((c, d) for c, d, _, _ in first.values())
    return {k: (c, d if seen[(c, d)] == 1 else f"{d}-{slug.rsplit('-', 1)[-1].lower()}", folder)
            for k, (c, d, slug, folder) in first.items()}


def _plain(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def planned_key(path: str, folders: dict[str, tuple[str, str, str]], by_dir: dict[str, set[str]],
                ids: dict[str, str]) -> str:
    """Clean bucket name for a dataset file that is not a catalogue render or video.
    It joins a design's files/ folder when it sits in that design's own folder, or
    deeper with the design id in its name (measured 2026-10-07: 20k files; a file
    deeper down without the id belonged to another design, so it goes to library/)."""
    d = os.path.dirname(path)
    while d and d not in by_dir:
        d = os.path.dirname(d)
    if d and len(by_dir[d]) == 1:
        key = next(iter(by_dir[d]))
        rel = path[len(d) + 1:]
        if "/" not in rel or (_plain(ids[key]) and _plain(ids[key]) in _plain(rel)):
            cat, folder, _ = folders[key]
            return ds.design_files_key(cat, folder, rel)
    return ds.library_key(path)


def plan_layout(man: ds.Manifest, rows: list[tuple]) -> int:
    """A clean, unique bucket name for every dataset file (storage.py: bucket layout).
    Names planned earlier are kept, so a name never changes once given."""
    have = dict(man.db.execute("SELECT path, key FROM layout").fetchall())
    used = set(have.values())
    folders = design_folders(rows)
    by_dir: dict[str, set[str]] = defaultdict(set)
    ids: dict[str, str] = {}
    want: list[tuple[str, str]] = []
    catalogue = set()
    for key, did, slug, cat, metal, view, path in rows:
        by_dir[os.path.dirname(path)].add(key)
        ids[key] = did
        c, folder, _ = folders[key]
        want.append((path, ds.catalogue_key(c, folder, metal, view, path.rsplit("/", 1)[-1])))
        catalogue.add(path)
    for (path,) in man.db.execute("SELECT path FROM scan UNION SELECT path FROM files ORDER BY 1").fetchall():
        if path not in catalogue:
            want.append((path, planned_key(path, folders, by_dir, ids)))
    new = []
    for path, key in want:
        if path in have:
            continue
        k, n = key, 2
        while k in used:
            k, n = ds.numbered(key, n), n + 1
        used.add(k)
        have[path] = k
        new.append((path, k))
    with man.write() as con:
        con.executemany("INSERT INTO layout VALUES (?,?)", new)
    return len(new)


def source_meta(rel: str) -> dict:
    """The dataset name travels with the object, so the bucket explains itself."""
    return {"source": quote(rel, safe="/ ()@#,-_.")[:1800]}


def put_original(s3: ds.S3Storage, src: Path, rel: str, size: int, key: str) -> dict:
    """One dataset file into the bucket under its clean name: zstd when it helps, else
    as it is, read from the SSD once where possible. Returns what the storage index records."""
    bucket = s3.bucket
    extra = {"ContentType": ds.content_type(rel), "Metadata": source_meta(rel)}
    if s3.originals_class:
        extra["StorageClass"] = s3.originals_class
    if ds.compressible(rel) and size >= 4096:
        from compression import zstd
        ds.TMP.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".zst", dir=ds.TMP)
        os.close(fd)
        tmp = Path(tmp)
        try:
            sha = hashlib.sha256()
            with open(src, "rb") as fin, zstd.open(tmp, "wb", level=ZSTD_LEVEL) as fout:
                while chunk := fin.read(8 * MiB):
                    sha.update(chunk)
                    fout.write(chunk)
            packed = tmp.stat().st_size
            if packed <= (1 - ZSTD_MIN_SAVING) * size:
                key += ds.ZSTD_SUFFIX
                s3.client.upload_file(str(tmp), bucket, key, ExtraArgs={**extra, "ContentType": "application/zstd"},
                                      Config=transfer_config())
                return {"path": rel, "key": key, "sha": sha.hexdigest(), "size": size, "stored": packed, "codec": "zstd",
                        "ctype": "application/zstd"}
        finally:
            tmp.unlink(missing_ok=True)
        s3.client.upload_file(str(src), bucket, key, ExtraArgs=extra, Config=transfer_config())
        return {"path": rel, "key": key, "sha": sha.hexdigest(), "size": size, "stored": size, "codec": "",
                "ctype": extra["ContentType"]}
    with open(src, "rb") as f:
        reader = HashingReader(f)
        s3.client.upload_fileobj(reader, bucket, key, ExtraArgs=extra, Config=transfer_config())
    if reader.n != size:
        raise IOError(f"{rel} changed while it was uploaded ({size} -> {reader.n} bytes)")
    return {"path": rel, "key": key, "sha": reader.sha.hexdigest(), "size": size, "stored": size, "codec": "",
            "ctype": extra["ContentType"]}


def record_original(man: ds.Manifest, con, r: dict, s3: ds.S3Storage, mtime: float | None):
    man.add_object(con, r["key"], r["sha"], r["size"], r["stored"], r["codec"], r["ctype"], "original", s3.originals_class)
    man.add_file(con, r["path"], r["key"], r["size"], mtime)


def cmd_web(args):
    cfg = settings(args)
    s3 = open_s3(cfg)
    require_bucket(s3)
    man = s3.manifest
    src = source_of(args)
    designs = load_designs()
    rows = design_rows(designs)
    with man.write() as con:
        con.execute("DELETE FROM design_files")
        con.executemany("INSERT OR REPLACE INTO design_files VALUES (?,?,?,?,?,?,?)", rows)
    print(f"{len(designs)} designs, {len(rows)} renders and videos listed in the storage index")
    plan_layout(man, rows)
    names = dict(man.db.execute("SELECT path, key FROM layout").fetchall())

    done_web = {r[0] for r in man.db.execute("SELECT path FROM web")}
    done_files = {r[0] for r in man.db.execute("SELECT path FROM files")}
    # catalogue pictures are always served as the originals: WebP copies only on request
    renders = [(r[6], ds.web_key(r[3], r[2], r[4], r[5])) for r in rows if r[5] != 5 and r[6] not in done_web] \
        if getattr(args, "image_web_copies", False) else []
    only = category_paths(args, rows)
    videos = [(r[6], ds.web_key(r[3], r[2], r[4], None, video=True)) for r in rows if r[5] == 5
              and (only is None or r[6] in only)]
    if args.limit:
        renders, videos = renders[:args.limit], videos[:max(1, args.limit // 4)]
    room = room_left(args, man)
    if room is not None:
        sizes = dict(man.db.execute("SELECT path, size FROM scan"))

        def size_of(path):
            return sizes.get(path) or (src.root / path).stat().st_size

        encode = args.videos == "encode"
        videos, skipped = within(videos, room, lambda v: (0 if v[0] in done_files else size_of(v[0]))
                                 + (size_of(v[0]) if encode and v[0] not in done_web else 0))
        print(f"--max-gb {args.max_gb:g}: {fmt(room)} left; {skipped} videos wait for a bigger limit", flush=True)

    # 1. renders -> WebP copies, only with --image-web-copies (CPU work: one process per core)
    print(f"renders: {len(renders)} to convert to WebP", flush=True)
    before = after = n = 0
    t0 = time.time()
    pool_cls = ProcessPoolExecutor if args.workers > 1 else ThreadPoolExecutor   # 1: in this process (tests)
    with pool_cls(max(args.workers, 1)) as pool:
        futs = [pool.submit(_render_job, cfg, str(src.root), p, k) for p, k in renders]
        batch = []
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                print(f"  failed: {e}", flush=True)
                continue
            batch.append(r)
            before += r["size"]
            after += r["stored"]
            n += 1
            if len(batch) >= 200 or n == len(futs):
                with man.write() as con:
                    for b in batch:
                        man.add_object(con, b["key"], b["sha"], b["size"], b["stored"], "", "image/webp", "web")
                        man.add_web(con, b["path"], b["key"], "display", b["sha"], b["w"], b["h"])
                batch = []
                print(f"  {n}/{len(futs)}  PNG {fmt(before)} -> WebP {fmt(after)}  {time.time() - t0:.0f}s", flush=True)

    # 2. videos: the original (served as it is) and, with --videos encode, a smaller H.264 web copy
    exe = ffmpeg_exe() if args.videos == "encode" else None
    if args.videos == "encode" and not exe:
        raise SystemExit("--videos encode needs ffmpeg (.venv/bin/pip install imageio-ffmpeg)")
    print(f"videos: {len(videos)} ({'re-encode to H.264' if exe else 'originals as they are'})", flush=True)
    t0 = time.time()
    lock = threading.Lock()

    def one_video(path, key):
        full = src.root / path
        size = full.stat().st_size
        out = {}
        if path not in done_files:
            out["original"] = put_original(s3, full, path, size, names[path])
        if exe and path not in done_web:
            enc = encode_video(exe, full)
            if enc:
                try:
                    small = enc.stat().st_size
                    if small <= (1 - VIDEO_MIN_SAVING) * size:
                        s3.client.upload_file(str(enc), cfg["bucket"], key, Config=transfer_config(), ExtraArgs={
                            "ContentType": "video/mp4", "CacheControl": WEB_CACHE_CONTROL})
                        known = man.file(path)
                        sha = (out.get("original") or {}).get("sha") or (known["sha256"] if known else sha256_file(full))
                        out["web"] = {"key": key, "size": size, "stored": small, "sha": sha}
                finally:
                    enc.unlink(missing_ok=True)
        return path, full.stat().st_mtime, out

    vb = va = 0
    full, stop = False, threading.Event()
    with ThreadPoolExecutor(args.video_workers if exe else args.workers) as pool:
        futs = [pool.submit(stop_when_full(one_video, stop), p, k) for p, k in videos]
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.cancelled():
                continue
            try:
                path, mtime, out = fut.result()
            except Exception as e:
                if bucket_full(e):
                    if not full:
                        full = True
                        for f in futs:
                            f.cancel()   # the ones not started; running ones finish and are recorded
                        print("  the bucket is full: stopping", flush=True)
                else:
                    print(f"  failed: {e}", flush=True)
                continue
            with lock, man.write() as con:
                if "original" in out:
                    record_original(man, con, out["original"], s3, mtime)
                if "web" in out:
                    w = out["web"]
                    man.add_object(con, w["key"], w["sha"], w["size"], w["stored"], "", "video/mp4", "web")
                    man.add_web(con, path, w["key"], "video", w["sha"])
                    vb += w["size"]
                    va += w["stored"]
            if i % 20 == 0 or i == len(futs):
                extra = f"  re-encoded {fmt(vb)} -> {fmt(va)}" if vb else ""
                print(f"  {i}/{len(futs)}{extra}  {time.time() - t0:.0f}s", flush=True)
    if full:
        raise SystemExit(BUCKET_FULL)
    print("web step done. The app can now run on S3: set JEWEL_STORAGE=s3://" + cfg["bucket"] + " in .env and restart.")


# ---------------------------------------------------------------- originals

def cmd_originals(args):
    cfg = settings(args)
    s3 = open_s3(cfg)
    require_bucket(s3)
    man = s3.manifest
    src = source_of(args)
    scan = man.db.execute("SELECT s.path, s.size, s.mtime, s.sha256 FROM scan s "
                          "LEFT JOIN files f ON f.path = s.path WHERE f.path IS NULL ORDER BY s.path").fetchall()
    if not man.db.execute("SELECT 1 FROM scan LIMIT 1").fetchone():
        raise SystemExit("Run the scan step first.")
    todo = [dict(r) for r in scan]
    try:
        rows = design_rows(load_designs())
    except Exception:
        rows = []
    catalogue = {r[6]: r[5] for r in rows}
    planned = plan_layout(man, rows)
    if planned:
        print(f"clean bucket names planned for {planned} files", flush=True)
    names = dict(man.db.execute("SELECT path, key FROM layout").fetchall())
    only = category_paths(args, rows)
    if only is not None:
        todo = [r for r in todo if r["path"] in only]
        print(f"--category {args.category}: only its catalogue pictures and videos", flush=True)
    prio = move_priority(catalogue)
    todo.sort(key=lambda r: prio(r["path"]))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} files to move ({fmt(sum(r['size'] for r in todo))})", flush=True)

    # identical files: only files whose size matches another file can be copies,
    # so only those are hashed up front; the rest are hashed while they upload
    stored_sizes = {r[0] for r in man.db.execute("SELECT DISTINCT size FROM objects WHERE tier='original'")}
    sizes = Counter(r["size"] for r in todo)
    need = [r for r in todo if r["sha256"] is None and r["size"] > 0 and (sizes[r["size"]] > 1 or r["size"] in stored_sizes)]
    if need:
        print(f"hashing {len(need)} files that may be copies of each other", flush=True)
        with ThreadPoolExecutor(8) as pool:
            for r, sha in zip(need, pool.map(lambda r: sha256_file(src.root / r["path"]), need)):
                r["sha256"] = sha
        with man.write() as con:
            con.executemany("UPDATE scan SET sha256 = ? WHERE path = ?", [(r["sha256"], r["path"]) for r in need])

    first: dict[str, dict] = {}
    upload, later = [], []
    aliases = 0
    with man.write() as con:
        for r in todo:
            sha = r["sha256"]
            if sha:
                obj = man.object_by_sha(sha)
                if obj:                                   # already in the bucket: point at it
                    man.add_file(con, r["path"], obj["key"], r["size"], r["mtime"])
                    aliases += 1
                    continue
                if sha in first:                          # a copy of a file in this run
                    later.append(r)
                    continue
                first[sha] = r
            upload.append(r)
    print(f"{aliases} copies of files already in the bucket recorded without uploading", flush=True)
    room = room_left(args, man)
    if room is not None:
        upload, skipped = within(upload, room, lambda r: r["size"])
        print(f"--max-gb {args.max_gb:g}: {len(upload)} files ({fmt(sum(r['size'] for r in upload))}) fit in the "
              f"{fmt(room)} left, pictures first; {skipped} files wait for a bigger limit", flush=True)

    t0, done, sent, stored = time.time(), 0, 0, 0
    big = threading.Semaphore(1)   # one very large file at a time (RAM)

    def job(r):
        full = src.root / r["path"]
        if r["size"] > 512 * MiB:
            with big:
                return r, put_original(s3, full, r["path"], r["size"], names[r["path"]])
        return r, put_original(s3, full, r["path"], r["size"], names[r["path"]])

    full, stop = False, threading.Event()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(stop_when_full(job, stop), r) for r in upload]
        for fut in as_completed(futs):
            if fut.cancelled():
                continue
            try:
                r, res = fut.result()
            except Exception as e:
                if bucket_full(e):
                    if not full:
                        full = True
                        for f in futs:
                            f.cancel()   # the ones not started; running ones finish and are recorded
                        print("  the bucket is full: stopping", flush=True)
                else:
                    print(f"  failed: {e}", flush=True)
                continue
            if r["sha256"] and r["sha256"] != res["sha"]:
                print(f"  warning: {r['path']} changed since it was hashed", flush=True)
            with man.write() as con:
                record_original(man, con, res, s3, r["mtime"])
            done += 1
            sent += res["size"]
            stored += res["stored"]
            if done % 200 == 0 or done == len(futs):
                rate = sent / max(time.time() - t0, 1)
                print(f"  {done}/{len(futs)}  {fmt(sent)} read, {fmt(stored)} stored  {fmt(rate)}/s", flush=True)

    with man.write() as con:
        for r in later:
            obj = man.object_by_sha(r["sha256"])
            if obj:
                man.add_file(con, r["path"], obj["key"], r["size"], r["mtime"])
    if full:
        raise SystemExit(BUCKET_FULL)
    print(f"{len(later)} identical copies stored once. originals step done.")


# ---------------------------------------------------------------- relayout

def empty_trash(s3: ds.S3Storage, workers: int = 8) -> int:
    """Delete, for good (every version), the old names of renamed objects."""
    man = s3.manifest
    keys = [r[0] for r in man.db.execute("SELECT key FROM trash").fetchall()]

    def one(key):
        remove_every_version(s3.client, s3.bucket, key)
        return key
    with ThreadPoolExecutor(workers) as pool:
        for key in pool.map(one, keys):
            with man.write() as con:
                con.execute("DELETE FROM trash WHERE key = ?", (key,))
    return len(keys)


def cmd_relayout(args):
    """Files already in the bucket get their clean names: a copy inside the bucket
    (nothing is downloaded or sent again), checked, recorded, then the old name is
    deleted with all its versions. Safe to stop and run again. Run it while the app is
    stopped: the app hands out links to the old names for up to 45 minutes."""
    from boto3.s3.transfer import TransferConfig
    cfg = settings(args)
    s3 = open_s3(cfg)
    require_bucket(s3)
    man = s3.manifest
    try:
        rows = design_rows(load_designs())
    except Exception:
        rows = []
    plan_layout(man, rows)
    prio = move_priority({r[6]: r[5] for r in rows})
    names = dict(man.db.execute("SELECT path, key FROM layout").fetchall())
    objs = man.db.execute("SELECT o.key, o.codec, o.content_type, o.stored, o.storage_class, "
                          "group_concat(f.path, char(0)) AS paths FROM objects o JOIN files f ON f.key = o.key "
                          "WHERE o.tier = 'original' GROUP BY o.key").fetchall()
    todo = []
    for o in objs:
        first = min(o["paths"].split("\0"), key=prio)
        new = names[first] + (ds.ZSTD_SUFFIX if o["codec"] == "zstd" else "")
        if new != o["key"]:
            todo.append((dict(o), first, new))
    if getattr(args, "limit", 0):
        todo = todo[:args.limit]
    print(f"{len(todo)} files to rename ({fmt(sum(o['stored'] for o, _, _ in todo))}) inside the bucket", flush=True)
    one_copy = TransferConfig(multipart_threshold=4 * 1024 * MiB, multipart_chunksize=512 * MiB)

    def job(item):
        o, first, new = item
        extra = {"ContentType": o["content_type"] or ds.content_type(first), "Metadata": source_meta(first),
                 "MetadataDirective": "REPLACE"}
        if o["storage_class"]:
            extra["StorageClass"] = o["storage_class"]
        s3.client.copy({"Bucket": s3.bucket, "Key": o["key"]}, s3.bucket, new, ExtraArgs=extra, Config=one_copy)
        size = s3.client.head_object(Bucket=s3.bucket, Key=new)["ContentLength"]
        if size != o["stored"]:
            raise IOError(f"{new}: {size} bytes after the copy, expected {o['stored']}")
        return item

    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(job, t) for t in todo]):
            try:
                o, _, new = fut.result()
            except Exception as e:
                failed += 1
                print(f"  failed: {e}", flush=True)
                continue
            with man.write() as con:
                con.execute("UPDATE objects SET key = ? WHERE key = ?", (new, o["key"]))
                con.execute("UPDATE files SET key = ? WHERE key = ?", (new, o["key"]))
                con.execute("INSERT OR IGNORE INTO trash VALUES (?)", (o["key"],))
            done += 1
            if done % 500 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} renamed  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    removed = empty_trash(s3, args.workers)
    print(f"{removed} old names deleted (all versions)", flush=True)
    if failed:
        raise SystemExit(f"{failed} files could not be renamed: run relayout again.")
    if getattr(args, "limit", 0):
        print(f"trial done: {done} files renamed; run relayout without --limit for the rest.")
    else:
        print("relayout done: every file in the bucket has its clean name.")


# ---------------------------------------------------------------- verify

def cmd_verify(args):
    cfg = settings(args)
    s3 = open_s3(cfg)
    man = s3.manifest
    objs = [dict(r) for r in man.db.execute("SELECT key, stored, size, sha256, codec, tier FROM objects WHERE tier != 'index'")]
    print(f"checking {len(objs)} objects are in the bucket with the right size", flush=True)
    problems = []

    def head(o):
        try:
            r = s3.client.head_object(Bucket=cfg["bucket"], Key=o["key"])
            return o, None if r["ContentLength"] == o["stored"] else f"size {r['ContentLength']} != {o['stored']}"
        except Exception as e:
            return o, f"missing ({type(e).__name__})"

    with ThreadPoolExecutor(16) as pool:
        for i, (o, err) in enumerate(pool.map(head, objs), 1):
            if err:
                problems.append((o["key"], err))
            if i % 5000 == 0:
                print(f"  {i}/{len(objs)}", flush=True)
    orphans = man.db.execute("SELECT COUNT(*) FROM files f LEFT JOIN objects o ON o.key = f.key WHERE o.key IS NULL").fetchone()[0]
    if orphans:
        problems.append(("storage index", f"{orphans} files point at no object"))
    bad = {k for k, _ in problems}
    originals = [o for o in objs if o["tier"] == "original" and o["key"] not in bad]
    sample = random.Random().sample(originals, min(args.sample, len(originals)))
    print(f"downloading {len(sample)} random originals to check their SHA-256", flush=True)
    for o in sample:
        try:
            data = s3.client.get_object(Bucket=cfg["bucket"], Key=o["key"])["Body"].read()
            if o["codec"] == "zstd":
                from compression import zstd
                data = zstd.decompress(data)
        except Exception as e:
            problems.append((o["key"], f"could not be read back ({type(e).__name__})"))
            continue
        if hashlib.sha256(data).hexdigest() != o["sha256"]:
            problems.append((o["key"], "content differs from the SSD"))
    for k, err in problems[:50]:
        print(f"  PROBLEM {k}: {err}")
    if problems:
        # take the broken entries out of the storage index, so the next run of
        # the web / originals step sends those files again
        keys = [(k,) for k, _ in problems if k != "storage index"]
        with man.write() as con:
            con.executemany("DELETE FROM web WHERE key = ?", keys)
            con.executemany("DELETE FROM files WHERE key = ?", keys)
            con.executemany("DELETE FROM objects WHERE key = ?", keys)
            con.execute("DELETE FROM files WHERE key NOT IN (SELECT key FROM objects)")
        raise SystemExit(f"{len(problems)} problems, taken out of the storage index. "
                         "Run the web and originals steps again to send those files again, then verify.")
    print("OK: everything in the storage index is in the bucket, and the sample matches the SSD byte for byte.")


# ---------------------------------------------------------------- index snapshots

SEARCH_BACKUP = ["index", "crops", "catalog.jsonl", "catalog_summary.json", "cad/specs.json",
                 "tryon/designs.json", "tryon/fidelity.json", "tryon/models", "audit"]
# never: bodyphotos/ (personal photos stay on this server), session_secret, logs, storage/


def cmd_index(args):
    cfg = settings(args)
    s3 = open_s3(cfg)
    man = s3.manifest
    designs = load_designs()
    rows = design_rows(designs)
    with man.write() as con:
        con.execute("DELETE FROM design_files")
        con.executemany("INSERT OR REPLACE INTO design_files VALUES (?,?,?,?,?,?,?)", rows)
    ds.TMP.mkdir(parents=True, exist_ok=True)
    snap = ds.TMP / "manifest.snapshot.sqlite"
    snap.unlink(missing_ok=True)
    dst = sqlite3.connect(snap)
    man.db.backup(dst)
    dst.close()
    s3.client.upload_file(str(snap), cfg["bucket"], ds.INDEX + "manifest.sqlite",
                          ExtraArgs={"ContentType": "application/vnd.sqlite3"})
    print(f"storage index snapshot uploaded ({fmt(snap.stat().st_size)})")
    snap.unlink()
    if args.no_search_backup:
        return
    tar = ds.TMP / f"search-{date.today():%Y%m%d}.tar.zst"
    with tarfile.open(tar, "w:zst") as t:
        for rel in SEARCH_BACKUP:
            p = DATA / rel
            if p.exists():
                t.add(p, arcname=f"data/{rel}")
    s3.client.upload_file(str(tar), cfg["bucket"], ds.INDEX + tar.name, Config=transfer_config(),
                          ExtraArgs={"ContentType": "application/zstd"})
    print(f"search index backup uploaded as {ds.INDEX + tar.name} ({fmt(tar.stat().st_size)})")
    tar.unlink()


def cmd_connect(args):
    """After buying the storage: check it, set it up, and switch the app to it.
    Files not moved yet keep coming from the SSD until `finish`."""
    cfg = settings(args)
    print("== 1/4 check the bucket and keys")
    cmd_check(args)
    print("\n== 2/4 set up the bucket")
    args.versioning = True
    cmd_setup(args)
    print("\n== 3/4 storage index")
    s3 = open_s3(cfg)
    if not s3.ensure_manifest():
        raise SystemExit("could not read or start the storage index")
    current = os.environ.get("JEWEL_STORAGE", "")
    fallback = os.environ.get("JEWEL_STORAGE_FALLBACK", "") or (current if current.startswith("file://") else "")
    args.no_search_backup = True
    cmd_index(args)
    print("\n== 4/4 switch the app (.env)")
    set_env({"JEWEL_STORAGE": f"s3://{cfg['bucket']}", "S3_BUCKET": cfg["bucket"],
             "JEWEL_STORAGE_FALLBACK": fallback or None})
    print(f"JEWEL_STORAGE=s3://{cfg['bucket']}" + (f"\nJEWEL_STORAGE_FALLBACK={fallback}  (until `finish`)" if fallback else ""))
    if fallback and not ds.from_url(fallback).ready():
        print("note: the SSD is not plugged in: files not moved yet can't be shown until it is")
    print("\nDone. Restart the app (scripts/start_live.sh) to use the bucket. Then start the move:\n"
          "  nohup .venv/bin/python scripts/migrate_to_s3.py move > data/storage/move.log 2>&1 &")


def cmd_move(args):
    """Everything from the SSD into the bucket, then verified. Safe to stop and start again."""
    src = source_of(args)
    args.source = str(src.root)
    require_bucket(open_s3(settings(args)))
    t0 = time.time()
    steps = [("scan the SSD", cmd_scan), ("clean names for files already in the bucket", cmd_relayout),
             ("display copies and videos", cmd_web),
             ("all originals", cmd_originals), ("verify", cmd_verify), ("index snapshots", cmd_index)]
    if getattr(args, "max_gb", 0) or getattr(args, "category", ""):
        # partial: the originals step sends pictures first; videos then get what is left
        steps[2], steps[3] = ("originals up to the limit, pictures first", cmd_originals), \
            ("videos with what is left", cmd_web)
    for title, step in steps:
        print(f"\n== {title}  ({(time.time() - t0) / 3600:.1f} h so far)", flush=True)
        step(args)
    left = ds.Manifest().db.execute("SELECT COUNT(*), COALESCE(SUM(s.size),0) FROM scan s LEFT JOIN files f "
                                    "ON f.path = s.path WHERE f.path IS NULL").fetchone()
    if left[0]:
        print(f"\nPart of the dataset is in the bucket and verified; {left[0]} files ({fmt(left[1])}) are still "
              "only on the SSD, and the app keeps reading those from there. Run `move` again with a bigger "
              "--max-gb, or without it, to send the rest.")
    else:
        print("\nThe move is complete and verified. Run `migrate_to_s3.py finish` to stop using the SSD.")


def cmd_finish(args):
    """Stop using the SSD: only when every scanned file is in the bucket and verify passes."""
    man = ds.Manifest()
    if not man.db.execute("SELECT 1 FROM scan LIMIT 1").fetchone():
        raise SystemExit("The SSD was never scanned: run `move` first.")
    pending = man.db.execute("SELECT COUNT(*) FROM scan s LEFT JOIN files f ON f.path = s.path "
                             "WHERE f.path IS NULL").fetchone()[0]
    if pending:
        raise SystemExit(f"{pending} files are not in the bucket yet: run `move` again first.")
    unservable = man.db.execute("SELECT COUNT(*) FROM design_files d LEFT JOIN web w ON w.path = d.path "
                                "LEFT JOIN files f ON f.path = d.path LEFT JOIN objects o ON o.key = f.key "
                                "WHERE w.path IS NULL AND (o.key IS NULL OR o.codec != '')").fetchone()[0]
    if unservable:
        raise SystemExit(f"{unservable} catalogue renders or videos can't be shown from the bucket yet: run `move` again.")
    cmd_verify(args)
    set_env({"JEWEL_STORAGE_FALLBACK": None})
    print("JEWEL_STORAGE_FALLBACK removed. Restart the app: it now uses only the bucket, never the SSD.\n"
          "Keep the SSD unchanged as an offline backup for a while.")


def cmd_restore(args):
    """A new or rebuilt server: the storage index and the app's search index from the bucket."""
    cfg = settings(args)
    s3 = open_s3(cfg)
    if s3.manifest.exists() and not args.force:
        print("storage index: already here")
    else:
        s3.manifest.path.unlink(missing_ok=True)
        print("storage index: " + ("downloaded" if s3.ensure_manifest() else "NOT FOUND in the bucket"))
    if (INDEX / "meta.jsonl").exists() and not args.force:
        print("search index: already here (use --force to replace it)")
        return
    r = s3.client.list_objects_v2(Bucket=cfg["bucket"], Prefix=ds.INDEX + "search-")
    backups = sorted(o["Key"] for o in r.get("Contents", []))
    if not backups:
        raise SystemExit("No search backup in the bucket (made by the index step).")
    ds.TMP.mkdir(parents=True, exist_ok=True)
    tar = ds.TMP / backups[-1].rsplit("/", 1)[-1]
    s3.client.download_file(cfg["bucket"], backups[-1], str(tar), Config=transfer_config())
    with tarfile.open(tar, "r:zst") as t:
        members = [m for m in t.getmembers() if m.name == "data" or m.name.startswith("data/")]
        t.extractall(DATA.parent, members=members, filter="data")   # "data" filter: nothing lands outside
    tar.unlink()
    print(f"search index restored from {backups[-1]}")


def cmd_status(args):
    man = ds.Manifest()
    s = man.stats()
    pending = man.db.execute("SELECT COUNT(*), COALESCE(SUM(s.size),0) FROM scan s LEFT JOIN files f ON f.path = s.path "
                             "WHERE f.path IS NULL").fetchone()
    print(f"SSD scanned            {s['ssd_scanned_files']} files, {fmt(s['ssd_scanned_bytes'])}")
    print(f"in the storage index   {s['dataset_files']} files, {fmt(s['dataset_bytes'])}")
    print(f"still to move          {pending[0]} files, {fmt(pending[1])}")
    print(f"identical copies       {s['duplicate_files']} files stored once")
    print(f"originals in bucket    {s['original_objects']} objects, {fmt(s['original_bytes'])} -> {fmt(s['original_stored_bytes'])} stored")
    if s["zstd_objects"]:
        print(f"  of which zstd        {s['zstd_objects']} files, {fmt(s['zstd_bytes_before'])} -> {fmt(s['zstd_bytes_after'])}")
    print(f"web copies             {s['web_objects']} objects, {fmt(s['web_bytes'])}")
    print(f"design files indexed   {s['design_files']}")
    for kind, v in s["by_kind"].items():
        print(f"  {kind:<9} {v['files']:>7} files  {fmt(v['bytes'] or 0):>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default="")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("connect")
    p = sub.add_parser("move")
    p.add_argument("--source", default="")
    p.add_argument("--videos", choices=["original", "encode"], default="original")
    p.add_argument("--image-web-copies", action="store_true",
                   help="also make WebP copies of the renders (never shown: pages show the originals)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--video-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-gb", type=float, default=0, help="partial move: stop when the bucket holds this many GB (pictures first)")
    p.add_argument("--category", default="", help="partial move: only this category (ring, earrings, bracelet, pendant, necklace)")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--no-search-backup", action="store_true")
    p = sub.add_parser("finish")
    p.add_argument("--sample", type=int, default=500)
    p = sub.add_parser("restore")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("setup")
    p.add_argument("--versioning", action="store_true", help="keep old versions of changed or deleted files")
    p = sub.add_parser("scan")
    p.add_argument("--source", default="", help="the SSD folder (default: from JEWEL_STORAGE(_FALLBACK) in .env)")
    p.add_argument("--estimate", action="store_true", help="compress a random sample to estimate the savings")
    p = sub.add_parser("web")
    p.add_argument("--source", default="", help="the SSD folder (default: from JEWEL_STORAGE(_FALLBACK) in .env)")
    p.add_argument("--videos", choices=["original", "encode"], default="original")
    p.add_argument("--image-web-copies", action="store_true",
                   help="also make WebP copies of the renders (never shown: pages show the originals)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--video-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only this many renders (a trial run)")
    p.add_argument("--max-gb", type=float, default=0, help="partial move: stop when the bucket holds this many GB (pictures first)")
    p.add_argument("--category", default="", help="partial move: only this category (ring, earrings, bracelet, pendant, necklace)")
    p = sub.add_parser("originals")
    p.add_argument("--source", default="", help="the SSD folder (default: from JEWEL_STORAGE(_FALLBACK) in .env)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-gb", type=float, default=0, help="partial move: stop when the bucket holds this many GB (pictures first)")
    p.add_argument("--category", default="", help="partial move: only this category (ring, earrings, bracelet, pendant, necklace)")
    p = sub.add_parser("relayout")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="only this many files (a trial run)")
    p = sub.add_parser("verify")
    p.add_argument("--sample", type=int, default=200)
    p = sub.add_parser("index")
    p.add_argument("--no-search-backup", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()
    {"check": cmd_check, "setup": cmd_setup, "scan": cmd_scan, "web": cmd_web, "originals": cmd_originals,
     "verify": cmd_verify, "relayout": cmd_relayout, "index": cmd_index, "status": cmd_status, "connect": cmd_connect, "move": cmd_move,
     "finish": cmd_finish, "restore": cmd_restore}[args.cmd](args)


if __name__ == "__main__":
    main()
