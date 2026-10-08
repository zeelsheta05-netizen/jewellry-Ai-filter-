"""Admin uploads of new design files straight into the dataset storage.

Files get the dataset name  New Dataset/<Category>/[<batch>/][<sub folders>/]<file name>
with their original names (the names carry design id, metal and view). Any file
type is accepted. Nothing in the storage is ever overwritten: a clashing name
gets " (2)", " (3)", ... appended.

Uploads are chunked so large files (3D videos, CAD) fit through the Cloudflare
tunnel's 100 MB request limit and a dropped connection only resends one chunk:

    start(...)                -> upload id
    append(id, offset, body)  -> repeat until every byte has arrived;
                                 the last chunk puts the file in place.

Where the bytes wait depends on the storage (jewelsearch/storage.py): in S3 they
are parts of a multipart upload, which only becomes a file when complete; in a
folder they go to New Dataset/.incoming/<id>.part and are renamed at the end.
Either way a half-sent file never appears in a category folder.
"""
import asyncio
import json
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import storage
from .config import DATA

NEW_DIR = "New Dataset"
# Folder names contain the keywords scripts/build_catalog.py looks for, so a
# later re-index files these designs under the right type.
FOLDERS = {"ring": "Rings", "earrings": "Earrings", "pendant": "Pendants",
           "necklace": "Necklaces", "bracelet": "Bracelets"}
OTHER = "other"   # the admin names the category folder (e.g. "Mangalsutra", "Nose Pins")
CHUNK = 32 * 1024 * 1024
MAX_FILE = 20 * 1024 ** 3
RESERVE = 5 * 1024 ** 3          # a dataset folder always keeps this much free (S3 has no limit)
IDLE = 2 * 3600                  # forget uploads with no chunk for this long
MAX_ACTIVE = 400
LOG = DATA / "uploads.log"

_BAD_CHARS = re.compile(r'[\x00-\x1f\x7f"*/:<>?\\|]')   # not allowed on exFAT / Windows
_JUNK = {"thumbs.db", "desktop.ini", ".ds_store"}


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def new_root() -> str:
    """Where uploads go, as the admin panel shows it."""
    return f"{storage.get().root_label()}/{NEW_DIR}"


def drive_ready() -> bool:
    """The storage answers (a folder must be mounted: never fill the internal disk by mistake)."""
    return storage.get().ready()


def clean_segment(name: str, what: str = "File name") -> str:
    s = _BAD_CHARS.sub("_", unicodedata.normalize("NFC", name).strip()).rstrip(". ")
    if not s:
        raise UploadError(400, f"{what} is empty.")
    if s.startswith(".") or s.startswith("~$") or s.lower() in _JUNK:
        raise UploadError(400, f"{what} '{name}' is a system/hidden file.")
    if len(s) > 200:
        raise UploadError(400, f"{what} is too long (max 200 characters).")
    return s


def custom_categories() -> list[str]:
    """Category folders admins created with "Other" (everything but the built-in five)."""
    builtin = {f.lower() for f in FOLDERS.values()}
    if not drive_ready():
        return []
    return sorted((d for d in storage.get().subdirs(NEW_DIR) if d.lower() not in builtin), key=str.lower)


def category_folder(category: str, custom: str = "") -> str:
    if category in FOLDERS:
        return FOLDERS[category]
    if category != OTHER:
        raise UploadError(400, "Choose a category.")
    name = clean_segment(custom.replace("\\", "/").strip().strip("/").split("/")[-1] if custom.strip() else "", "Category name")
    # reuse an existing folder's spelling (the SSD was case-insensitive, so "rings" is "Rings")
    for existing in [*FOLDERS.values(), *custom_categories()]:
        if existing.lower() == name.lower():
            return existing
    return name


def batch_parts(folder: str, batch: str) -> list[str]:
    """The admin's own folder name(s), without any storage path typed or pasted in front.

    "s3://bucket/New Dataset/Pendants/AA", "/Volumes/Storage/New Dataset/Pendants/AA",
    "New Dataset/Pendants/AA" and "AA" all give ["AA"]; "AA/Set 1" gives ["AA", "Set 1"].
    """
    segs = [s.strip() for s in batch.replace("\\", "/").split("/") if s.strip()]
    if segs and segs[0].lower() in ("s3:", "file:"):
        segs = segs[1:]
    root = [*storage.get().root_parts(), NEW_DIR]
    for k in range(len(root)):          # longest matching tail of the root path first
        tail = [s.lower() for s in root[k:]]
        if [s.lower() for s in segs[:len(tail)]] == tail:
            segs = segs[len(tail):]
            break
    if segs and segs[0].lower() == folder.lower():
        segs = segs[1:]
    elif segs and segs[0].lower() in {f.lower() for f in FOLDERS.values()}:
        raise UploadError(400, f"Folder path points to '{segs[0]}' but the category is {folder}.")
    return [clean_segment(s, "Folder name") for s in segs]


def target_dir(category: str, batch: str = "", subdirs: str = "", custom: str = "") -> str:
    """The dataset name of the folder the files go to ("New Dataset/Rings/Loat - 15")."""
    folder = category_folder(category, custom)
    parts = [folder, *batch_parts(folder, batch)]
    dirs = [p for p in subdirs.replace("\\", "/").split("/") if p.strip()]
    if len(dirs) > 8:
        raise UploadError(400, "Folder is nested too deeply (max 8 levels).")
    parts += [clean_segment(p, "Folder name") for p in dirs]
    if any(p in (".", "..") for p in parts):
        raise UploadError(400, "Invalid folder.")
    return "/".join([NEW_DIR, *parts])


def _split(name: str) -> tuple[str, str]:
    i = name.rfind(".")
    return (name[:i], name[i:]) if i > 0 else (name, "")


def unique_path(path: str) -> str:
    st = storage.get()
    if not st.exists(path):
        return path
    d, _, name = path.rpartition("/")
    stem, ext = _split(name)
    i = 2
    while True:
        p = f"{d}/{stem} ({i}){ext}"
        if not st.exists(p):
            return p
        i += 1


def unique_dir(path: str) -> str:
    """Like unique_path, for folders ("Loat - 1.5" has no extension to keep)."""
    st, base, i = storage.get(), path, 2
    while st.is_dir(path):
        path = f"{base} ({i})"
        i += 1
    return path


def existing_folders(category: str, custom: str, rels: list[str]) -> list[dict]:
    """Which of these folders (relative to the category folder) already exist in the
    storage, how many files they hold, and a free name to use instead."""
    if not drive_ready():
        raise UploadError(503, "The dataset storage is not connected.")
    st = storage.get()
    base = f"{NEW_DIR}/{category_folder(category, custom)}"
    out = []
    for rel in dict.fromkeys(rels):
        d = target_dir(category, rel, "", custom)
        if d != base and st.is_dir(d):
            out.append({"rel": rel, "path": d, "files": st.count_files(d),
                        "suggestion": unique_dir(d)[len(base) + 1:]})
    return out


@dataclass
class Upload:
    id: str
    owner: str
    email: str
    name: str
    size: int
    handle: storage.UploadHandle
    received: int = 0
    touched: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_active: dict[str, Upload] = {}
_last_sweep = [0.0]


def _sweep():
    now = time.monotonic()
    st = storage.get()
    for uid, up in list(_active.items()):
        if now - up.touched > IDLE and not up.lock.locked():
            _active.pop(uid, None)
            st.abort_upload(up.handle)
    # leftovers from before a server restart (at most every 10 minutes: in S3 it is a listing)
    if now - _last_sweep[0] > 600:
        _last_sweep[0] = now
        live = {str(up.handle.part) if up.handle.part else up.handle.upload_id for up in _active.values()}
        try:
            st.sweep_uploads(NEW_DIR, live, IDLE)
        except Exception:
            pass


def free_bytes() -> int | None:
    """None: no space limit (S3)."""
    return storage.get().free_bytes() if drive_ready() else 0


def _pending() -> int:
    return sum(up.size - up.received for up in _active.values())


def start(owner: str, email: str, category: str, name: str, size: int, batch: str = "", subdirs: str = "",
          custom: str = "") -> Upload:
    if not drive_ready():
        raise UploadError(503, "The dataset storage is not connected.")
    _sweep()
    name = clean_segment(name)
    dest_dir = target_dir(category, batch, subdirs, custom)
    if not 0 <= size <= MAX_FILE:
        raise UploadError(413, f"Files must be under {MAX_FILE // 1024 ** 3} GB.")
    if len(_active) >= MAX_ACTIVE:
        raise UploadError(429, "Too many uploads in progress. Wait for some to finish.")
    free = free_bytes()
    if free is not None and free - _pending() - size < RESERVE:
        raise UploadError(507, "Not enough free space in the dataset storage.")
    st = storage.get()
    for _ in range(5):   # the free name is reserved at once, so two uploads can't pick the same one
        try:
            handle = st.begin_upload(unique_path(f"{dest_dir}/{name}"), size)
            break
        except FileExistsError:
            continue
    else:
        raise UploadError(409, "Could not find a free file name. Try again.")
    uid = secrets.token_urlsafe(16)
    up = Upload(uid, owner, email, name, size, handle)
    _active[uid] = up
    return up


async def append(upload_id: str, owner: str, offset: int, body) -> dict:
    """Write one chunk (an async iterator of bytes) at `offset`. Returns progress;
    after the last byte, also where the file was saved."""
    up = _active.get(upload_id)
    if not up or up.owner != owner:
        raise UploadError(404, "Upload not found or expired. Start it again.")
    st = storage.get()
    async with up.lock:
        if offset != up.received:
            raise UploadError(409, f"Expected offset {up.received}.")
        buf = bytearray()   # one chunk (at most 32 MB) is gathered, then stored in one go
        async for piece in body:
            if len(buf) + len(piece) > CHUNK or up.received + len(buf) + len(piece) > up.size:
                raise UploadError(413, "Chunk is larger than expected.")
            buf += piece
        last = up.received + len(buf) == up.size
        try:
            if last:
                await asyncio.to_thread(_finish, st, up.handle, bytes(buf))
            elif buf:
                await asyncio.to_thread(st.write_upload, up.handle, bytes(buf))
        except FileExistsError:
            _active.pop(upload_id, None)
            raise UploadError(409, "A file with this name appeared meanwhile. Upload it again.")
        except storage.StorageUnavailable:
            raise UploadError(503, "The dataset storage did not answer. The chunk will be sent again.")
        up.received += len(buf)
        up.touched = time.monotonic()
        if not last:
            return {"received": up.received, "done": False}
        _active.pop(upload_id, None)
        rel = up.handle.path
        _log(up, rel)
        return {"received": up.received, "done": True, "path": rel, "renamed": rel.rsplit("/", 1)[-1] != up.name}


def _finish(st: storage.Storage, h: storage.UploadHandle, data: bytes):
    """The last chunk and putting the file in place succeed or fail together:
    on failure nothing has changed, so the page can send the same chunk again."""
    saved = (h.written, list(h.parts), h.buf, h.sha.copy())
    try:
        if data:
            st.write_upload(h, data)
        st.commit_upload(h)
    except Exception:
        h.written, h.parts, h.buf, h.sha = saved
        raise


def cancel(upload_id: str, owner: str):
    up = _active.get(upload_id)
    if up and up.owner == owner and not up.lock.locked():
        _active.pop(upload_id, None)
        storage.get().abort_upload(up.handle)


def _log(up: Upload, rel: str):
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": up.email,
             "path": rel, "size": up.size}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def recent(limit: int = 100) -> list[dict]:
    if not LOG.is_file():
        return []
    lines = LOG.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out
