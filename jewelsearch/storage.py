"""Where the dataset lives, and the only way the app reads or writes it.

The dataset is addressed by URL, never by a disk path. One setting picks it:

    JEWEL_STORAGE=s3://<bucket>              the S3 bucket (AWS S3 or an S3-compatible service)
    JEWEL_STORAGE=file:///Volumes/Storage    a folder: the old SSD, or a test folder

Every dataset file keeps its dataset name, for example
"01/01/Loat - 02/Ring/DDLR-423/1174@Y-#viwe1.png". That name is an ID, not a
place on disk. This module turns it into:

  media_url(name)    a short-lived signed https URL the browser loads directly
  read_bytes(name)   the file's bytes, for the server's own processing
  local_copy(name)   a cached local file, for tools that need a real file (CAD)

Bucket layout (clean, readable names; the dataset name is kept in the index and
in each object's "source" metadata):

    catalogue/<category>/<design>/<metal>/view-<N>.png  the renders the app shows, unchanged
    catalogue/<category>/<design>/<metal>/video.mp4     the 3D video, unchanged
    catalogue/<category>/<design>/files/...             that design's CAD, extra pictures and
                                                        job cards (found in the design's folder)
    library/<category>/[not-uploaded/]lot-<N>/...       everything else from the SSD, by category
                                                        and lot, with tidy folder and file names
    library/mixed/lot-<N>/...                           a lot's shared folders (photos, xlsx) that
                                                        hold several categories
    uploads/<category>/...                              files added through the admin upload page
    web/<category>/<design slug>/<metal>/...            smaller web copies (videos; WebP only with
                                                        --image-web-copies)
    index/manifest.sqlite                               snapshot of the storage index
    index/search-<date>.tar.zst                         snapshot of the app's search index

Every file is stored once, unchanged. Files that compress well (CAD, text) are
stored zstd-compressed (".zst" added) when that saves >= 10 %.

The storage index ("manifest", data/storage/manifest.sqlite) answers every
"where is this file / which files does this design have / what is in this
folder" question with one indexed lookup, so nothing ever lists or scans the
bucket. scripts/migrate_to_s3.py fills it; admin uploads add to it.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlsplit

from .config import DATA

STATE = DATA / "storage"
MANIFEST = STATE / "manifest.sqlite"
CACHE = STATE / "cache"          # local copies of originals (CAD files, job cards), size-capped
TMP = STATE / "tmp"

ORIGINALS = "originals/"     # old layout (dataset name as the key); `relayout` renames these
CATALOGUE = "catalogue/"
LIBRARY = "library/"
UPLOADS = "uploads/"
WEB = "web/"
INDEX = "index/"
ZSTD_SUFFIX = ".zst"

URL_TTL = 3600          # signed links work for an hour ...
URL_REUSE = 45 * 60     # ... and the same link is handed out for 45 min, so the browser cache works
PART_SIZE = 16 * 1024 * 1024   # S3 multipart part size for admin uploads (S3 minimum is 5 MiB)

# files and folders that are never data (OS litter on the SSD)
JUNK_DIRS = {"$RECYCLE.BIN", ".Spotlight-V100", ".Trashes", ".fseventsd", "System Volume Information"}
JUNK_FILES = {"thumbs.db", "desktop.ini", ".ds_store"}

KINDS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"},
    "video": {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"},
    "cad": {".3dm", ".stl", ".obj", ".jcd", ".step", ".stp", ".igs", ".iges", ".ply", ".fbx", ".dxf"},
    "jobcard": {".xlsx", ".xls", ".csv"},
    "document": {".pdf", ".doc", ".docx", ".txt"},
}
# already compressed: zstd would only waste time
NO_COMPRESS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
               ".zip", ".rar", ".7z", ".gz", ".xz", ".zst", ".bz2", ".xlsx", ".docx", ".pptx", ".pdf", ".mp3", ".aac"}


class StorageUnavailable(Exception):
    """The storage can't be reached (bucket offline, drive unplugged, no credentials)."""


def is_junk(name: str) -> bool:
    return name.startswith("._") or name.startswith("~$") or name.lower() in JUNK_FILES


def kind_of(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return next((k for k, exts in KINDS.items() if ext in exts), "other")


def compressible(name: str) -> bool:
    return os.path.splitext(name)[1].lower() not in NO_COMPRESS


def content_type(name: str) -> str:
    if name.endswith(ZSTD_SUFFIX):
        return "application/zstd"
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def check_name(path: str) -> str:
    """A dataset name must stay inside the dataset: no "..", no absolute paths."""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise FileNotFoundError(path)
    if any(p in ("", ".", "..") for p in path.split("/")):
        raise FileNotFoundError(path)
    return path


def design_slug(design_id: str, first_folder: str) -> str:
    """Same slug as the try-on models (scripts/build_tryon_models.py): bare ids
    such as "14" repeat across batches, so the folder makes it unique."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", design_id).strip("_")[:60] or "design"
    return f"{base}-{hashlib.sha1(f'{design_id}|{first_folder}'.encode()).hexdigest()[:6]}"


def web_key(category: str, slug: str, metal: str, view: int | None, video: bool = False) -> str:
    name = "turntable.mp4" if video else f"view{view}.webp"
    return f"{WEB}{category or 'unknown'}/{slug}/{metal}/{name}"


# ---------------------------------------------------------------- bucket names

# the SSD's misspelt folder words, fixed in bucket names only (dataset names stay as they are)
WORD_FIXES = {"loat": "lot", "ledis": "ladies", "brelcate": "bracelet", "neckless": "necklace",
              "pandant": "pendant", "viwe": "view"}
# folder words -> category; "earring" before "ring"
CATEGORY_WORDS = [("earring", "earrings"), ("brelcate", "bracelet"), ("bracelet", "bracelet"),
                  ("neckless", "necklace"), ("necklace", "necklace"), ("pandant", "pendant"),
                  ("pendant", "pendant"), ("ring", "ring"), ("chain", "chain")]
# a folder that only names the category ("EARRINGS", "rings", "BRELCATE NOT UPLOAD") adds nothing
CATEGORY_FOLDERS = {"ring", "rings", "earring", "earrings", "bracelet", "bracelets", "necklace", "necklaces",
                    "pendant", "pendants"}


def clean_part(s: str) -> str:
    """One readable bucket-name part: lower case, a-z 0-9 . _ -, common misspellings fixed."""
    s = s.strip().lower().replace("'", "")
    s = re.sub(r"[^a-z0-9._]+", "-", s)
    s = re.sub(r"[a-z]+", lambda m: WORD_FIXES.get(m.group(), m.group()), s)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return s or "x"


def clean_name(name: str) -> str:
    stem, ext = os.path.splitext(name)
    if not stem:                                   # ".hidden" style names
        stem, ext = name, ""
    ext = re.sub(r"[^a-z0-9]+", "", ext.lower())
    return clean_part(stem) + ("." + ext if ext else "")


def category_of(dirs: list[str]) -> str:
    for d in dirs:
        low = d.lower()
        for word, cat in CATEGORY_WORDS:
            if word in low:
                return cat
    return "other"


def tidy_dirs(dirs: list[str]) -> list[str]:
    """Clean folder names, without category-only folders and repeated folders ("x/x/x.stl")."""
    out = []
    for d in dirs:
        c = clean_part(d)
        bare = re.sub(r"-?not-upload(ed)?$", "", c)
        if bare in CATEGORY_FOLDERS or (out and out[-1] == c):
            continue
        out.append(c)
    return out


def library_key(path: str) -> str:
    """Bucket name for a dataset file that is not part of a catalogue design:
    library/<category>/[not-uploaded/]<lot>/<folders>/<file>, or uploads/... for
    files added through the upload page ("New Dataset/...")."""
    *dirs, name = path.split("/")
    while dirs and re.fullmatch(r"\d+", dirs[0].strip()):       # the SSD's "01/01/" batch folders
        dirs.pop(0)
    root = LIBRARY
    if dirs and dirs[0] == "New Dataset":
        root, dirs = UPLOADS, dirs[1:]
    cat, tidy = category_of(dirs), tidy_dirs(dirs)
    if cat == "other" and tidy and re.fullmatch(r"lot-\d+", tidy[0]):
        cat = "mixed"                   # a lot's shared folders (photos, xlsx) hold several categories
    parts = [cat]
    if any("not upload" in d.lower() for d in dirs):
        parts.append("not-uploaded")
    parts += tidy
    stem = clean_name(name)
    return root + "/".join(parts + [stem])


def catalogue_key(category: str, design: str, metal: str, view: int, name: str) -> str:
    ext = clean_name(name).rpartition(".")[2] if "." in name else ""
    file = (f"view-{view}" if view != 5 else "video") + (f".{ext}" if ext else "")
    return f"{CATALOGUE}{clean_part(category or 'other')}/{design}/{clean_part(metal).replace('_', '-')}/{file}"


def design_files_key(category: str, design: str, rel: str) -> str:
    """A design's other files (CAD, extra pictures), rel = path inside the design's folder."""
    *dirs, name = rel.split("/")
    return f"{CATALOGUE}{clean_part(category or 'other')}/{design}/files/" + "/".join(tidy_dirs(dirs) + [clean_name(name)])


def numbered(key: str, n: int) -> str:
    """"a/b.png" -> "a/b-2.png": a second file that would get the same name."""
    d, _, name = key.rpartition("/")
    stem, ext = os.path.splitext(name)
    return f"{d}/{stem}-{n}{ext}"


def zstd_decompress_file(src: Path, dst: Path):
    from compression import zstd
    with zstd.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)


# ---------------------------------------------------------------- the storage index

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (            -- what is stored in the bucket
    key TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,                       -- of the original bytes
    size INTEGER NOT NULL,                      -- original bytes
    stored INTEGER NOT NULL,                    -- bytes in the bucket, after compression
    codec TEXT NOT NULL DEFAULT '',             -- '' or 'zstd'
    content_type TEXT,
    tier TEXT NOT NULL,                         -- 'original' | 'web' | 'index'
    storage_class TEXT,
    uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS objects_sha ON objects(sha256, tier);
CREATE TABLE IF NOT EXISTS files (              -- every dataset name -> the object holding its bytes
    path TEXT PRIMARY KEY,
    key TEXT NOT NULL,                          -- duplicates point at the first copy's object
    dir TEXT NOT NULL,
    name TEXT NOT NULL,
    ext TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL
);
CREATE INDEX IF NOT EXISTS files_dir ON files(dir);
CREATE INDEX IF NOT EXISTS files_ext ON files(ext);
CREATE INDEX IF NOT EXISTS files_key ON files(key);
CREATE TABLE IF NOT EXISTS web (                -- smaller copies the browser is given
    path TEXT PRIMARY KEY,                      -- dataset name of the source file
    key TEXT NOT NULL,
    kind TEXT NOT NULL,                         -- 'display' (WebP) | 'video' (re-encoded MP4)
    source_sha256 TEXT NOT NULL,
    width INTEGER,
    height INTEGER
);
CREATE TABLE IF NOT EXISTS design_files (       -- catalogue designs -> their renders and videos
    design_key TEXT NOT NULL,
    design_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    category TEXT NOT NULL,
    metal TEXT NOT NULL,
    view INTEGER NOT NULL,                      -- 1-4 renders, 5 = 3D video
    path TEXT NOT NULL,
    PRIMARY KEY (design_key, metal, view)
);
CREATE INDEX IF NOT EXISTS design_files_id ON design_files(design_id);
CREATE INDEX IF NOT EXISTS design_files_cat ON design_files(category, metal);
CREATE INDEX IF NOT EXISTS design_files_path ON design_files(path);
CREATE TABLE IF NOT EXISTS scan (               -- the SSD as last scanned by the migration
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha256 TEXT
);
CREATE INDEX IF NOT EXISTS scan_size ON scan(size);
CREATE TABLE IF NOT EXISTS layout (             -- the clean bucket name planned for each dataset file
    path TEXT PRIMARY KEY,
    key TEXT NOT NULL UNIQUE                    -- without ".zst"
);
CREATE TABLE IF NOT EXISTS trash (              -- old bucket names still to delete after a rename
    key TEXT PRIMARY KEY
);
"""


def _upper_bound(prefix: str) -> str:
    """Smallest string after every string that starts with `prefix`."""
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


class Manifest:
    """The storage index: one SQLite file, safe to read from many threads and
    processes while the migration writes to it (WAL mode)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or MANIFEST)
        self._local = threading.local()

    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def db(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.path, timeout=30)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
            con.executescript(SCHEMA)
            self._local.con = con
        return con

    @contextmanager
    def write(self):
        con = self.db
        with con:
            yield con

    def close(self):
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None

    # ---- lookups
    def file(self, path: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT f.path, f.key, f.size, f.kind, o.codec, o.stored, o.sha256 FROM files f "
            "JOIN objects o ON o.key = f.key WHERE f.path = ?", (path,)).fetchone()

    def web_copy(self, path: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT key, kind, width, height FROM web WHERE path = ?", (path,)).fetchone()

    def object_by_sha(self, sha256: str, tier: str = "original") -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM objects WHERE sha256 = ? AND tier = ? LIMIT 1",
                               (sha256, tier)).fetchone()

    def known(self, path: str) -> bool:
        """In the bucket, or on the SSD waiting to be moved: either way the name is taken."""
        return self.db.execute("SELECT 1 FROM files WHERE path = ? UNION ALL SELECT 1 FROM scan WHERE path = ? LIMIT 1",
                               (path, path)).fetchone() is not None

    def paths(self, prefix: str = "", exts: set[str] | None = None) -> Iterator[str]:
        if prefix:
            prefix = prefix.rstrip("/") + "/"
            rows = self.db.execute("SELECT path, ext FROM files WHERE path >= ? AND path < ? ORDER BY path",
                                   (prefix, _upper_bound(prefix)))
        else:
            rows = self.db.execute("SELECT path, ext FROM files ORDER BY path")
        for r in rows.fetchall():
            if exts is None or r["ext"] in exts:
                yield r["path"]

    def subdirs(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        rows = self.db.execute("SELECT DISTINCT dir FROM files WHERE dir >= ? AND dir < ?",
                               (prefix, _upper_bound(prefix))).fetchall()
        names = {r["dir"][len(prefix):].split("/", 1)[0] for r in rows if r["dir"]}
        return sorted(n for n in names if n and not n.startswith("."))

    def count(self, prefix: str, cap: int = 100_000) -> int:
        prefix = prefix.rstrip("/") + "/"
        return self.db.execute("SELECT COUNT(*) FROM (SELECT 1 FROM files WHERE path >= ? AND path < ? LIMIT ?)",
                               (prefix, _upper_bound(prefix), cap)).fetchone()[0]

    def layout_key(self, path: str) -> str | None:
        r = self.db.execute("SELECT key FROM layout WHERE path = ?", (path,)).fetchone()
        return r[0] if r else None

    def key_taken(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM layout WHERE key = ? UNION ALL SELECT 1 FROM objects WHERE key = ? "
                               "OR key = ? LIMIT 1", (key, key, key + ZSTD_SUFFIX)).fetchone() is not None

    def design(self, design_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM design_files WHERE design_id = ? ORDER BY design_key, metal, view",
                               (design_id,)).fetchall()

    # ---- writes
    def add_object(self, con, key: str, sha256: str, size: int, stored: int, codec: str, ctype: str, tier: str,
                   storage_class: str = ""):
        con.execute("INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                    (key, sha256, size, stored, codec, ctype, tier, storage_class))

    def add_file(self, con, path: str, key: str, size: int, mtime: float | None = None):
        d, _, name = path.rpartition("/")
        con.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)",
                    (path, key, d, name, os.path.splitext(name)[1].lower(), kind_of(name), size, mtime))

    def add_web(self, con, path: str, key: str, kind: str, source_sha256: str, width=None, height=None):
        con.execute("INSERT OR REPLACE INTO web VALUES (?,?,?,?,?,?)", (path, key, kind, source_sha256, width, height))

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()   # noqa: E731
        files = q("SELECT COUNT(*), COALESCE(SUM(size),0) FROM files")
        orig = q("SELECT COUNT(*), COALESCE(SUM(size),0), COALESCE(SUM(stored),0) FROM objects WHERE tier='original'")
        web = q("SELECT COUNT(*), COALESCE(SUM(stored),0) FROM objects WHERE tier='web'")
        zst = q("SELECT COUNT(*), COALESCE(SUM(size),0), COALESCE(SUM(stored),0) FROM objects WHERE codec='zstd'")
        by_kind = {r[0]: {"files": r[1], "bytes": r[2]} for r in
                   self.db.execute("SELECT kind, COUNT(*), SUM(size) FROM files GROUP BY kind ORDER BY 3 DESC")}
        scan = q("SELECT COUNT(*), COALESCE(SUM(size),0) FROM scan")
        return {
            "dataset_files": files[0], "dataset_bytes": files[1],
            "original_objects": orig[0], "original_bytes": orig[1], "original_stored_bytes": orig[2],
            "duplicate_files": files[0] - q("SELECT COUNT(DISTINCT key) FROM files")[0],
            "zstd_objects": zst[0], "zstd_bytes_before": zst[1], "zstd_bytes_after": zst[2],
            "web_objects": web[0], "web_bytes": web[1],
            "design_files": q("SELECT COUNT(*) FROM design_files")[0],
            "by_kind": by_kind, "ssd_scanned_files": scan[0], "ssd_scanned_bytes": scan[1],
        }


# ---------------------------------------------------------------- uploads (admin panel)

@dataclass
class UploadHandle:
    path: str                     # dataset name the file will have
    size: int
    written: int = 0
    # local folder
    part: Path | None = None
    # S3
    key: str = ""
    upload_id: str | None = None
    parts: list = field(default_factory=list)
    buf: bytes = b""
    sha: "hashlib._Hash" = field(default_factory=hashlib.sha256)


# ---------------------------------------------------------------- backends

class Storage:
    url: str
    _reserved: set[str]

    def label(self) -> str: ...
    def ready(self) -> bool: ...
    def media_origin(self) -> str | None: return None
    def media_url(self, path: str) -> str | None: return None
    def read_bytes(self, path: str, prefer: str = "original") -> bytes: ...
    def local_copy(self, path: str) -> Path: ...
    def exists(self, path: str) -> bool: ...
    def size(self, path: str) -> int | None: ...
    def walk(self, prefix: str = "", exts: set[str] | None = None) -> Iterator[str]: ...
    def subdirs(self, prefix: str) -> list[str]: ...
    def count_files(self, prefix: str, cap: int = 100_000) -> int: ...
    def free_bytes(self) -> int | None: return None
    def root_parts(self) -> list[str]: ...
    def root_label(self) -> str: ...
    def is_dir(self, prefix: str) -> bool: ...
    def name_of(self, local: Path) -> str | None: return None   # dataset name of a local_copy() file
    def begin_upload(self, path: str, size: int) -> UploadHandle: ...
    def write_upload(self, h: UploadHandle, data: bytes): ...
    def commit_upload(self, h: UploadHandle): ...
    def abort_upload(self, h: UploadHandle): ...
    def sweep_uploads(self, prefix: str, live: set[str], idle: float): ...


class LocalStorage(Storage):
    """A folder (the SSD before the move, and the tests)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.url = self.root.as_uri()
        self._place = threading.Lock()
        self._reserved: set[str] = set()

    def label(self) -> str:
        return f"dataset folder {self.root}"

    def ready(self) -> bool:
        """Mounted (never create folders on the internal disk by mistake when the SSD is unplugged)."""
        if not self.root.is_dir():
            return False
        return os.path.ismount(self.root) if str(self.root).startswith("/Volumes/") else True

    def _file(self, path: str) -> Path:
        f = self.root / check_name(path)
        if not f.is_file():
            if not self.ready():
                raise StorageUnavailable(f"{self.label()} is not connected")
            raise FileNotFoundError(path)
        return f

    def read_bytes(self, path: str, prefer: str = "original") -> bytes:
        return self._file(path).read_bytes()

    def local_copy(self, path: str) -> Path:
        return self._file(path)

    def exists(self, path: str) -> bool:
        return path in self._reserved or (self.root / check_name(path)).exists()

    def is_dir(self, prefix: str) -> bool:
        return (self.root / check_name(prefix)).is_dir()

    def size(self, path: str) -> int | None:
        try:
            return (self.root / check_name(path)).stat().st_size
        except OSError:
            return None

    def walk(self, prefix: str = "", exts: set[str] | None = None) -> Iterator[str]:
        base = self.root / prefix if prefix else self.root
        stack = [base]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError:
                continue
            for e in entries:
                if e.name in JUNK_DIRS or is_junk(e.name) or e.name.startswith("."):
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif exts is None or os.path.splitext(e.name)[1].lower() in exts:
                    yield str(Path(e.path).relative_to(self.root))

    def subdirs(self, prefix: str) -> list[str]:
        d = self.root / prefix
        if not self.ready() or not d.is_dir():
            return []
        return sorted(x.name for x in d.iterdir() if x.is_dir() and not x.name.startswith("."))

    def count_files(self, prefix: str, cap: int = 100_000) -> int:
        n = 0
        for _dir, subdirs, files in os.walk(self.root / prefix):
            subdirs[:] = [s for s in subdirs if not s.startswith(".")]
            n += sum(1 for f in files if not f.startswith("."))
            if n >= cap:
                break
        return n

    def free_bytes(self) -> int | None:
        return shutil.disk_usage(self.root).free if self.ready() else 0

    def root_parts(self) -> list[str]:
        return list(self.root.parts[1:])

    def root_label(self) -> str:
        return str(self.root)

    def name_of(self, local: Path) -> str | None:
        local = Path(local)
        return str(local.relative_to(self.root)) if local.is_relative_to(self.root) else None

    def _incoming(self, path: str) -> Path:
        return self.root / path.split("/", 1)[0] / ".incoming"

    def begin_upload(self, path: str, size: int) -> UploadHandle:
        inc = self._incoming(check_name(path))
        with self._place:
            if self.exists(path):
                raise FileExistsError(path)
            self._reserved.add(path)
        inc.mkdir(parents=True, exist_ok=True)
        part = inc / f"{hashlib.sha1(f'{path}{time.time_ns()}'.encode()).hexdigest()[:20]}.part"
        part.touch()
        return UploadHandle(path, size, part=part)

    def write_upload(self, h: UploadHandle, data: bytes):
        with open(h.part, "r+b") as f:
            f.truncate(h.written)       # drop bytes of a chunk that failed half-way
            f.seek(h.written)
            f.write(data)
            if h.written + len(data) == h.size:
                f.flush()
                os.fsync(f.fileno())
        h.written += len(data)

    def commit_upload(self, h: UploadHandle):
        dest = self.root / h.path
        with self._place:
            if dest.exists():
                self._reserved.discard(h.path)
                raise FileExistsError(h.path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(h.part, dest)
            self._reserved.discard(h.path)

    def abort_upload(self, h: UploadHandle):
        self._reserved.discard(h.path)
        if h.part:
            h.part.unlink(missing_ok=True)

    def sweep_uploads(self, prefix: str, live: set[str], idle: float):
        inc = self.root / prefix / ".incoming"
        if not inc.is_dir():
            return
        for p in inc.glob("*.part"):
            try:
                if str(p) not in live and time.time() - p.stat().st_mtime > idle:
                    p.unlink()
            except OSError:
                pass


class S3Storage(Storage):
    """An S3 bucket. Reads go through the storage index; the browser gets
    signed links and downloads straight from the bucket."""

    def __init__(self, bucket: str, region: str | None = None, endpoint: str | None = None,
                 manifest: Manifest | None = None, cache_dir: Path | None = None, cache_bytes: int | None = None,
                 originals_class: str | None = None):
        self.bucket = bucket
        self.region = region or None
        self.endpoint = endpoint or None
        self.url = f"s3://{bucket}"
        self.manifest = manifest or Manifest()
        self.cache_dir = Path(cache_dir or CACHE)
        self.cache_bytes = cache_bytes if cache_bytes is not None else int(float(os.environ.get("STORAGE_CACHE_GB", "20")) * 1024 ** 3)
        # Intelligent-Tiering moves rarely used originals to cheaper tiers by itself, with no
        # retrieval fee or delay; S3-compatible services (R2, Wasabi ...) don't have it.
        default_class = "" if self.endpoint else "INTELLIGENT_TIERING"
        self.originals_class = originals_class if originals_class is not None else os.environ.get("S3_ORIGINALS_CLASS", default_class)
        self._client = None
        self._client_lock = threading.Lock()
        self._urls: dict[str, tuple[str, float]] = {}
        self._ready = (0.0, False)
        self._reserved: set[str] = set()
        self._reserved_keys: set[str] = set()     # bucket names of uploads in progress
        self._place = threading.Lock()

    # ---- connection
    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import boto3
                    from botocore.config import Config
                    # S3-compatible services (Cloudflare R2, Backblaze, Wasabi) accept different
                    # optional checksum headers: send them only where S3 requires one. Integrity
                    # is still checked end to end (TLS, sizes, SHA-256 in migrate_to_s3.py verify).
                    compat = {"request_checksum_calculation": "when_required",
                              "response_checksum_validation": "when_required"} if self.endpoint else {}
                    cfg = Config(signature_version="s3v4", retries={"max_attempts": 8, "mode": "adaptive"},
                                 max_pool_connections=32,
                                 s3={"addressing_style": "path" if self.endpoint else "virtual"}, **compat)
                    self._client = boto3.client("s3", region_name=self.region, endpoint_url=self.endpoint, config=cfg)
        return self._client

    def label(self) -> str:
        return f"S3 bucket {self.bucket}"

    def ready(self) -> bool:
        """The bucket answers and the storage index is here (checked at most once a minute)."""
        at, ok = self._ready
        if time.monotonic() - at < 60:
            return ok
        try:
            self.client.head_bucket(Bucket=self.bucket)
            ok = self.ensure_manifest()
        except Exception:
            ok = False
        self._ready = (time.monotonic(), ok)
        return ok

    def ensure_manifest(self) -> bool:
        """A new server (or a lost data folder) gets the storage index from the bucket's snapshot."""
        if self.manifest.exists():
            return True
        from botocore.exceptions import ClientError
        self.manifest.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest.path.with_suffix(".download")
        try:
            self.client.download_file(self.bucket, INDEX + "manifest.sqlite", str(tmp))
        except ClientError as e:
            tmp.unlink(missing_ok=True)
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                self.manifest.db   # a new bucket: start an empty index
                return True
            return False
        except Exception:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(self.manifest.path)
        return True

    def _error(self, e: Exception, path: str):
        from botocore.exceptions import BotoCoreError, ClientError
        if isinstance(e, ClientError):
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                raise FileNotFoundError(path) from e
            raise StorageUnavailable(f"{self.label()}: {code or e}") from e
        if isinstance(e, BotoCoreError):
            raise StorageUnavailable(f"{self.label()}: {e}") from e
        raise e

    # ---- reads
    def media_origin(self) -> str | None:
        u = urlsplit(self.client.generate_presigned_url("get_object", {"Bucket": self.bucket, "Key": "x"}, ExpiresIn=60))
        return f"{u.scheme}://{u.netloc}"

    def serving_key(self, path: str) -> str | None:
        """What the browser is given for a dataset file: a picture always as its original,
        unchanged; a video as its web copy when there is one; else the original (only if
        stored as is: a .zst can't be shown)."""
        web = None if kind_of(path) == "image" else self.manifest.web_copy(path)
        if web:
            return web["key"]
        f = self.manifest.file(path)
        return f["key"] if f and not f["codec"] else None

    def media_url(self, path: str) -> str | None:
        now = time.monotonic()
        hit = self._urls.get(path)
        if hit and now - hit[1] < URL_REUSE:
            return hit[0]
        key = self.serving_key(path)
        if key is None:
            raise FileNotFoundError(path)
        url = self.client.generate_presigned_url("get_object", {"Bucket": self.bucket, "Key": key}, ExpiresIn=URL_TTL)
        if len(self._urls) > 50_000:
            self._urls.clear()
        self._urls[path] = (url, now)
        return url

    def _get(self, key: str, path: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:
            self._error(e, path)

    def read_bytes(self, path: str, prefer: str = "original") -> bytes:
        f, web = self.manifest.file(path), self.manifest.web_copy(path)
        order = [("web", web), ("original", f)] if prefer == "web" else [("original", f), ("web", web)]
        for tier, row in order:
            if row is None:
                continue
            data = self._get(row["key"], path)
            if tier == "original" and row["codec"] == "zstd":
                from compression import zstd
                data = zstd.decompress(data)
            return data
        raise FileNotFoundError(path)

    def local_copy(self, path: str) -> Path:
        """The original as a local file, kept in a size-capped cache (oldest out first)."""
        f = self.manifest.file(check_name(path))
        if f is None:
            raise FileNotFoundError(path)
        dest = self.cache_dir / path
        if dest.is_file() and dest.stat().st_size == f["size"]:
            os.utime(dest)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            self.client.download_file(self.bucket, f["key"], str(tmp))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            self._error(e, path)
        if f["codec"] == "zstd":
            raw = tmp.with_name(tmp.name + ".raw")
            zstd_decompress_file(tmp, raw)
            tmp.unlink()
            tmp = raw
        tmp.replace(dest)
        self._evict(keep=dest)
        return dest

    def _evict(self, keep: Path):
        files = []
        for root, _dirs, names in os.walk(self.cache_dir):
            for n in names:
                p = Path(root) / n
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
        total = sum(s for _, s, _ in files)
        for mtime, size, p in sorted(files):
            if total <= self.cache_bytes:
                break
            if p == keep or time.time() - mtime < 600:   # never pull a file from under a running job
                continue
            p.unlink(missing_ok=True)
            total -= size

    def exists(self, path: str) -> bool:
        return path in self._reserved or self.manifest.known(check_name(path))

    def size(self, path: str) -> int | None:
        f = self.manifest.file(path)
        return f["size"] if f else None

    def walk(self, prefix: str = "", exts: set[str] | None = None) -> Iterator[str]:
        return self.manifest.paths(prefix, exts)

    def subdirs(self, prefix: str) -> list[str]:
        return self.manifest.subdirs(prefix)

    def count_files(self, prefix: str, cap: int = 100_000) -> int:
        return self.manifest.count(prefix, cap)

    def root_parts(self) -> list[str]:
        return [self.bucket]

    def root_label(self) -> str:
        return f"s3://{self.bucket}"

    def name_of(self, local: Path) -> str | None:
        local = Path(local)
        return str(local.relative_to(self.cache_dir)) if local.is_relative_to(self.cache_dir) else None

    def is_dir(self, prefix: str) -> bool:
        return self.manifest.count(check_name(prefix), 1) > 0

    # ---- admin uploads: straight into uploads/, as a multipart upload
    def begin_upload(self, path: str, size: int) -> UploadHandle:
        check_name(path)
        with self._place:
            if self.exists(path):
                raise FileExistsError(path)
            base, key, n = library_key(path), library_key(path), 2
            while key in self._reserved_keys or self.manifest.key_taken(key):
                key, n = numbered(base, n), n + 1
            self._reserved.add(path)
            self._reserved_keys.add(key)
        return UploadHandle(path, size, key=key)

    def _extra(self, h: UploadHandle) -> dict:
        extra = {"ContentType": content_type(h.path)}
        if self.originals_class:
            extra["StorageClass"] = self.originals_class
        return extra

    def write_upload(self, h: UploadHandle, data: bytes):
        """Parts are sent once 16 MiB have arrived. State changes only after every
        part of this chunk is in the bucket, so a failed chunk can simply be resent."""
        buf, pos, sent = h.buf + data, 0, []
        try:
            while len(buf) - pos >= PART_SIZE:
                if h.upload_id is None:
                    h.upload_id = self.client.create_multipart_upload(Bucket=self.bucket, Key=h.key, **self._extra(h))["UploadId"]
                n = len(h.parts) + len(sent) + 1
                r = self.client.upload_part(Bucket=self.bucket, Key=h.key, UploadId=h.upload_id, PartNumber=n,
                                            Body=buf[pos:pos + PART_SIZE])
                sent.append({"PartNumber": n, "ETag": r["ETag"]})
                pos += PART_SIZE
        except Exception as e:
            self._error(e, h.path)
        h.parts += sent
        h.buf = buf[pos:]
        h.sha.update(data)
        h.written += len(data)

    def commit_upload(self, h: UploadHandle):
        try:
            if h.upload_id is None:
                self.client.put_object(Bucket=self.bucket, Key=h.key, Body=h.buf, **self._conditional(), **self._extra(h))
            else:
                parts = list(h.parts)
                if h.buf or not parts:
                    n = len(parts) + 1
                    r = self.client.upload_part(Bucket=self.bucket, Key=h.key, UploadId=h.upload_id, PartNumber=n, Body=h.buf)
                    parts.append({"PartNumber": n, "ETag": r["ETag"]})
                self.client.complete_multipart_upload(Bucket=self.bucket, Key=h.key, UploadId=h.upload_id,
                                                      MultipartUpload={"Parts": parts}, **self._conditional())
        except Exception as e:
            from botocore.exceptions import ClientError
            if isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412"):
                self.abort_upload(h)
                raise FileExistsError(h.path) from e
            self._error(e, h.path)
        with self.manifest.write() as con:
            self.manifest.add_object(con, h.key, h.sha.hexdigest(), h.size, h.size, "", content_type(h.path),
                                     "original", self.originals_class)
            self.manifest.add_file(con, h.path, h.key, h.size, time.time())
            con.execute("INSERT OR REPLACE INTO layout VALUES (?,?)", (h.path, h.key))
        self._reserved.discard(h.path)
        self._reserved_keys.discard(h.key)

    def _conditional(self) -> dict:
        """Never replace an object that is already there (S3 conditional write).
        Off for services that don't support it: S3_CONDITIONAL_WRITES=0."""
        on = os.environ.get("S3_CONDITIONAL_WRITES", "0" if self.endpoint else "1") == "1"
        return {"IfNoneMatch": "*"} if on else {}

    def abort_upload(self, h: UploadHandle):
        self._reserved.discard(h.path)
        self._reserved_keys.discard(h.key)
        if h.upload_id:
            try:
                self.client.abort_multipart_upload(Bucket=self.bucket, Key=h.key, UploadId=h.upload_id)
            except Exception:
                pass   # the bucket's lifecycle rule removes leftovers after a few days

    def sweep_uploads(self, prefix: str, live: set[str], idle: float):
        """Multipart uploads left behind by a restart (also removed by the bucket's lifecycle rule)."""
        try:
            r = self.client.list_multipart_uploads(Bucket=self.bucket, Prefix=UPLOADS)
        except Exception:
            return
        for u in r.get("Uploads", []):
            started = u["Initiated"].timestamp()
            if u["UploadId"] not in live and time.time() - started > idle:
                try:
                    self.client.abort_multipart_upload(Bucket=self.bucket, Key=u["Key"], UploadId=u["UploadId"])
                except Exception:
                    pass


class UnconfiguredStorage(Storage):
    """JEWEL_STORAGE is missing from .env: search still works (it never reads
    the dataset); full-size media, try-on photos and uploads say so clearly."""

    url = ""

    def _no(self, *_a, **_k):
        raise StorageUnavailable("JEWEL_STORAGE is not set in .env")

    read_bytes = local_copy = begin_upload = write_upload = commit_upload = _no

    def label(self) -> str:
        return "dataset storage (JEWEL_STORAGE not set in .env)"

    def ready(self) -> bool:
        return False

    def media_url(self, path: str) -> str | None:
        self._no()

    def exists(self, path: str) -> bool:
        return False

    def size(self, path: str) -> int | None:
        return None

    def walk(self, prefix: str = "", exts: set[str] | None = None) -> Iterator[str]:
        return iter(())

    def subdirs(self, prefix: str) -> list[str]:
        return []

    def count_files(self, prefix: str, cap: int = 100_000) -> int:
        return 0

    def free_bytes(self) -> int | None:
        return 0

    def root_parts(self) -> list[str]:
        return []

    def root_label(self) -> str:
        return "(not set)"

    def is_dir(self, prefix: str) -> bool:
        return False

    def abort_upload(self, h: UploadHandle):
        pass

    def sweep_uploads(self, prefix: str, live: set[str], idle: float):
        pass


class FallbackStorage(Storage):
    """The bucket, with the SSD behind it while the move runs: reads try the
    bucket first and fall back to the SSD for files not moved yet; new files
    (admin uploads) go only to the bucket. Set by JEWEL_STORAGE_FALLBACK and
    removed by `migrate_to_s3.py finish` once everything is verified."""

    def __init__(self, primary: Storage, fallback: Storage):
        self.primary, self.fallback = primary, fallback
        self.url = primary.url

    def label(self) -> str:
        return f"{self.primary.label()} (with {self.fallback.label()} until the move is finished)"

    def ready(self) -> bool:
        return self.primary.ready()

    def media_origin(self) -> str | None:
        return self.primary.media_origin()

    def _either(self, method: str, path: str, *args, **kw):
        # not moved yet (FileNotFoundError), or the bucket didn't answer (StorageUnavailable:
        # network, provider trouble): the SSD still holds every file until `finish`
        try:
            return getattr(self.primary, method)(path, *args, **kw)
        except (FileNotFoundError, StorageUnavailable):
            if not self.fallback.ready():
                raise
            return getattr(self.fallback, method)(path, *args, **kw)

    def media_url(self, path: str) -> str | None:
        return self._either("media_url", path)          # None from the SSD: the app sends the file itself

    def read_bytes(self, path: str, prefer: str = "original") -> bytes:
        return self._either("read_bytes", path, prefer)

    def local_copy(self, path: str) -> Path:
        return self._either("local_copy", path)

    def exists(self, path: str) -> bool:
        return self.primary.exists(path) or (self.fallback.ready() and self.fallback.exists(path))

    def size(self, path: str) -> int | None:
        n = self.primary.size(path)
        return n if n is not None or not self.fallback.ready() else self.fallback.size(path)

    def walk(self, prefix: str = "", exts: set[str] | None = None) -> Iterator[str]:
        names = set(self.primary.walk(prefix, exts))
        if self.fallback.ready():
            names.update(self.fallback.walk(prefix, exts))
        return iter(sorted(names))

    def subdirs(self, prefix: str) -> list[str]:
        names = set(self.primary.subdirs(prefix))
        if self.fallback.ready():
            names.update(self.fallback.subdirs(prefix))
        return sorted(names)

    def count_files(self, prefix: str, cap: int = 100_000) -> int:
        return min(cap, sum(1 for _ in self.walk(prefix)))

    def free_bytes(self) -> int | None:
        return self.primary.free_bytes()

    def root_parts(self) -> list[str]:
        return self.primary.root_parts()

    def root_label(self) -> str:
        return self.primary.root_label()

    def is_dir(self, prefix: str) -> bool:
        return self.primary.is_dir(prefix) or (self.fallback.ready() and self.fallback.is_dir(prefix))

    def name_of(self, local: Path) -> str | None:
        return self.primary.name_of(local) or self.fallback.name_of(local)

    def begin_upload(self, path: str, size: int) -> UploadHandle:
        if self.fallback.ready() and self.fallback.exists(path):   # never shadow a file not moved yet
            raise FileExistsError(path)
        return self.primary.begin_upload(path, size)

    def write_upload(self, h: UploadHandle, data: bytes):
        self.primary.write_upload(h, data)

    def commit_upload(self, h: UploadHandle):
        self.primary.commit_upload(h)

    def abort_upload(self, h: UploadHandle):
        self.primary.abort_upload(h)

    def sweep_uploads(self, prefix: str, live: set[str], idle: float):
        self.primary.sweep_uploads(prefix, live, idle)


# ---------------------------------------------------------------- the configured storage

def from_url(url: str) -> Storage:
    u = urlsplit(url)
    if u.scheme == "s3":
        if not u.netloc:
            raise ValueError("JEWEL_STORAGE=s3://<bucket name>")
        return S3Storage(u.netloc, os.environ.get("S3_REGION") or os.environ.get("AWS_REGION"),
                         os.environ.get("S3_ENDPOINT"))
    if u.scheme == "file":
        return LocalStorage(unquote(u.path))
    raise ValueError(f"JEWEL_STORAGE must be s3://<bucket> or file:///<folder>, not {url!r}")


_current: Storage | None = None
_current_lock = threading.Lock()


def configured(url: str, fallback_url: str = "") -> Storage:
    if not url:
        return UnconfiguredStorage()
    st = from_url(url)
    if fallback_url and fallback_url != url:
        st = FallbackStorage(st, from_url(fallback_url))
    return st


def kind(st: Storage | None = None) -> str:
    """"s3", "folder" or "none": what the admin panel shows (a move in progress counts as s3)."""
    st = st or get()
    if isinstance(st, FallbackStorage):
        st = st.primary
    return "s3" if isinstance(st, S3Storage) else "folder" if isinstance(st, LocalStorage) else "none"


def get() -> Storage:
    global _current
    if _current is None:
        with _current_lock:
            if _current is None:
                from .config import STORAGE_FALLBACK_URL, STORAGE_URL
                _current = configured(STORAGE_URL, STORAGE_FALLBACK_URL)
    return _current


def use(s: Storage | None):
    """Switch storage (tests, scripts)."""
    global _current
    _current = s
