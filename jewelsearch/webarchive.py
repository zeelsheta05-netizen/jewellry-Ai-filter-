"""Every product page the team fetches is kept in the dataset storage
(storage.get(): the S3 bucket, or the dataset folder), whether it is listed
or not, so the scraped designs build up into a collection:

    Web Designs/<site>/<date>_<product slug>_<id>/
        01.jpg, 02.png, ...   the pictures, byte for byte as the shop served them, at the
                              largest size its image server gives (brands.download_pictures)
        preview.jpg           a small preview for the team panel (the only file made here)
        page.html             the product page as fetched
        product.json          the shop's product data (Shopify shops)
        design.json           everything read from the page: name, brand, SKU, price,
                              specification, tables, key figures, and each picture's
                              source link, size and SHA-256

Files are first written to data/web_designs/ on this Mac (the fetch answers
at once), then sent to the dataset storage in the background with the same
uploads as the admin panel, and the local copy is removed. If the storage
can't be reached they wait there and are sent with the next fetch.

Each fetch is also a row of public.web_designs (supabase/web_designs.sql):
the team panel lists the collection from it and can list any stored design
later without fetching the page again.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from . import storage
from .auth import _call
from .config import DATA

log = logging.getLogger(__name__)
WEB_DIR = "Web Designs"
STAGING = DATA / "web_designs"
TABLE = "/rest/v1/web_designs"
RECORD = ".record"          # local only: the web_designs row of a staged folder
RETRY_AFTER = 300           # a staged folder left behind is retried after this many seconds
LIST_FIELDS = ("id,url,site,title,brand,price,currency,folder,archive_status,pictures,brand_design_id,"
               "scraped_by_name,created_at")
_push_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _segment(text: str, limit: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:limit].strip("-")
    return s or "design"


def folder_for(original: dict) -> str:
    """ "Web Designs/melorra.com/2026-10-05_autumn-blues-diamond-rings_3f9a2c" """
    site = re.sub(r"[^A-Za-z0-9.-]+", "_", original.get("site") or "unknown-site").strip("._") or "unknown-site"
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{WEB_DIR}/{site}/{day}_{_segment(original.get('title', ''), 60)}_{secrets.token_hex(3)}"


def _local(folder: str) -> Path:
    path = STAGING / storage.check_name(folder)
    if not path.resolve().is_relative_to(STAGING.resolve()):
        raise FileNotFoundError(folder)
    return path


def stage(folder: str, pictures: list, original: dict, page: bytes, product: dict | None, scraped_by: str,
          preview_jpeg: bytes | None) -> list[dict]:
    """Writes the fetched page into data/web_designs/<folder>/ -> each picture's record."""
    d = _local(folder)
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, p in enumerate(pictures, 1):
        name = f"{i:02d}.{p.ext}"
        (d / name).write_bytes(p.data)
        files.append({"file": name, "width": p.width, "height": p.height, "bytes": len(p.data),
                      "sha256": hashlib.sha256(p.data).hexdigest(), "content_type": p.ctype,
                      "source": p.source, "asked": p.asked})
    if preview_jpeg:
        (d / "preview.jpg").write_bytes(preview_jpeg)
    if page:
        (d / "page.html").write_bytes(page)
    if product:
        (d / "product.json").write_text(json.dumps(product, ensure_ascii=False, indent=1))
    (d / "design.json").write_text(json.dumps({
        "source": "web", "url": original.get("url"), "site": original.get("site"), "folder": folder,
        "scraped_at": original.get("fetched_at"), "scraped_by": scraped_by, "original": original, "pictures": files,
        "about": "Pictures are the files the shop served, unchanged, at the largest size its image server gave. "
                 "preview.jpg is a small copy made for the team panel.",
    }, ensure_ascii=False, indent=1))
    return files


def set_record(folder: str, row_id: int):
    try:
        (_local(folder) / RECORD).write_text(str(row_id))
    except OSError:
        pass   # already sent and removed: the caller marks the row itself


def read(folder: str, name: str) -> bytes:
    """One file of a fetched page: the local copy while it waits, else the dataset storage."""
    try:
        return (_local(folder) / name).read_bytes()
    except FileNotFoundError:
        return storage.get().read_bytes(f"{folder}/{name}")


def local_file(folder: str, name: str) -> Path | None:
    p = _local(folder) / name
    return p if p.is_file() else None


def push(folder: str) -> bool:
    """Sends a staged folder to the dataset storage; True when all of it is there
    (the local copy is then removed). Never replaces a file already stored."""
    d = _local(folder)
    if not d.is_dir():
        return True
    st = storage.get()
    if not st.ready():
        return False
    with _push_lock:
        for f in sorted(p for p in d.iterdir() if p.is_file() and p.name != RECORD):
            name = f"{folder}/{f.name}"
            data = f.read_bytes()
            try:
                h = st.begin_upload(name, len(data))
            except FileExistsError:
                continue   # sent by an earlier try that stopped half-way
            except Exception as e:
                log.warning("web archive: %s not sent (%s)", name, e)
                return False
            try:
                st.write_upload(h, data)
                st.commit_upload(h)
            except FileExistsError:
                continue
            except Exception as e:
                st.abort_upload(h)
                log.warning("web archive: %s not sent (%s)", name, e)
                return False
        shutil.rmtree(d, ignore_errors=True)
        for parent in d.parents:   # empty site / date folders
            if parent == STAGING or not parent.is_relative_to(STAGING):
                break
            try:
                parent.rmdir()
            except OSError:
                break
    return True


def staged(older_than: float = 0) -> list[str]:
    """Folders still waiting on this Mac."""
    root = STAGING / WEB_DIR
    if not root.is_dir():
        return []
    now = time.time()
    return [f"{WEB_DIR}/{d.parent.name}/{d.name}" for d in root.glob("*/*")
            if d.is_dir() and (d / "design.json").is_file() and now - d.stat().st_mtime >= older_than]


def _record_id(folder: str) -> int | None:
    try:
        return int((_local(folder) / RECORD).read_text())
    except (OSError, ValueError):
        return None


async def push_and_mark(folder: str, row_id: int | None = None):
    """Background task after a fetch: this folder, then any left behind earlier."""
    for f in [folder] + [x for x in staged(RETRY_AFTER) if x != folder]:
        rid = row_id if f == folder else _record_id(f)
        rid = rid or _record_id(f)
        ok = await run_in_threadpool(push, f)
        if ok and rid:
            await update(rid, {"archive_status": "stored"})


# ---------------------------------------------------------------- the collection (Supabase)

class ArchiveError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


async def _req(method: str, params: dict | None = None, **kw):
    r = await _call(method, TABLE, params=params or {}, **kw)
    if r.status_code == 404:
        raise ArchiveError(503, "The scraped-designs list isn't set up yet (run supabase/web_designs.sql in Supabase). "
                                "Fetched pages are still stored in the dataset storage.")
    if r.status_code >= 300:
        log.warning("web designs: %s -> %s %s", method, r.status_code, r.text[:300])
        raise ArchiveError(502, "The scraped-designs list is unavailable right now.")
    return r


async def record(original: dict, folder: str, files: list[dict], user: dict) -> int | None:
    """A row for the fetched page. Best effort: without the table the files are still stored."""
    row = {"url": original["url"], "site": original["site"], "title": original.get("title", ""),
           "brand": original.get("brand", ""), "price": original.get("price"), "currency": original.get("currency", ""),
           "original": original, "pictures": files, "folder": folder, "archive_status": "staged",
           "scraped_by": user["uid"], "scraped_by_name": user.get("name") or user.get("email") or "",
           "updated_at": _now()}
    try:
        r = await _req("POST", json=row, headers={"Prefer": "return=representation"})
        return r.json()[0]["id"]
    except ArchiveError as e:
        log.warning("web designs: not recorded: %s", e.message)
        return None


async def update(row_id: int, changes: dict):
    try:
        await _req("PATCH", {"id": f"eq.{row_id}"}, json={**changes, "updated_at": _now()})
    except ArchiveError as e:
        log.warning("web designs %s: not updated: %s", row_id, e.message)


async def listing(limit: int = 60, q: str = "") -> list[dict]:
    params = {"select": LIST_FIELDS, "order": "created_at.desc", "limit": str(limit)}
    q = re.sub(r"[^\w .-]", "", q or "", flags=re.UNICODE).strip()[:60]
    if q:
        params["or"] = "(" + ",".join(f'{c}.ilike."*{q}*"' for c in ("title", "site", "brand", "url")) + ")"
    return (await _req("GET", params)).json()


async def get(row_id: int) -> dict:
    rows = (await _req("GET", {"id": f"eq.{row_id}", "select": "*"})).json()
    if not rows:
        raise ArchiveError(404, "This scraped design isn't in the collection.")
    return rows[0]


def where(row: dict) -> str:
    """Where a fetched page's files are, as the panel says it."""
    if row.get("archive_status") == "stored":
        return f"{storage.get().root_label()}/{row['folder']}"
    return f"waiting on this Mac (data/web_designs/{row['folder']}), sent to the dataset storage when it answers"
