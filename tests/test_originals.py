"""Catalogue pictures are the dataset's original renders, unchanged.

Cards, detail views, category tiles, history, favourites and orders link to
/media/<id>, which sends the dataset file exactly as it is. data/crops (small
cut-outs) is only what the image models read, and a stand-in when the
original can't be read. Try-on previews make their own images (tryon.py).
Run: .venv/bin/python -m pytest tests/test_originals.py -q
"""
import json

import pytest

from jewelsearch import storage
from jewelsearch.config import INDEX, media_token, thumb_name

needs_index = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")
RENDER = "01/Loat - 1/Ring/DDLR-1/DDLR-1@W-#viwe1.png"
VIDEO = "01/Loat - 1/Ring/DDLR-1/DDLR-1-02@W-#viwe5.mp4"


# ---- every link the pages get (index needed) -------------------------------------

@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine(judge=lambda t: "yes")


@needs_index
def test_every_catalogue_picture_is_the_original(engine):
    uid = next(u for u, m in engine.by_uid.items() if len(m["images"]) == 3)
    m = engine.by_uid[uid]
    card = engine.card(uid)
    for metal, url in card["thumbs_by_metal"].items():
        assert engine.media_by_token[url.rsplit("/", 1)[1]] == engine._front_path(m, metal)
    detail = engine.detail(uid)
    for views in detail["views"].values():
        assert all(v["thumb"] == v["full"] and v["full"].startswith("/media/") for v in views)
    shown = [card, detail, engine.categories(), engine.similar(uid), engine.browse("ring"), engine.search("rose gold rings")]
    assert "/crops/" not in json.dumps(shown)


@needs_index
def test_orders_keep_previews_but_show_originals(engine):
    from jewelsearch import server
    uid = next(iter(engine.by_uid))
    original = engine.card(uid)["thumb"]
    preview = engine.preview_url(original)
    assert preview.startswith("/crops/") and engine.original_url(preview) == original
    assert engine.original_url("/crops/0000000000000000.webp") == "/crops/0000000000000000.webp"   # left the catalogue
    server.engine, before = engine, server.engine
    try:
        order = {"thumb": preview, "snapshot": {"design": {"views": [preview, preview]}}}
        assert server._originals(order) == {"thumb": original, "snapshot": {"design": {"views": [original, original]}}}
    finally:
        server.engine = before


# ---- the /media route (no index needed) -------------------------------------------

@pytest.fixture
def media(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    ssd = tmp_path / "ssd"
    (ssd / RENDER).parent.mkdir(parents=True)
    (ssd / RENDER).write_bytes(b"\x89PNG original render bytes")
    crops = tmp_path / "crops"
    crops.mkdir()
    (crops / thumb_name(RENDER)).write_bytes(b"RIFF small preview")
    monkeypatch.setattr(server, "CROPS", crops)
    monkeypatch.setattr(server, "engine", type("E", (), {"media_by_token": {media_token(p): p for p in (RENDER, VIDEO)}})())
    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u1", "email": "a@b.co", "name": "A"})

    async def ok(uid, fresh=False):
        return True
    monkeypatch.setattr(auth, "is_approved", ok)
    before = storage._current
    yield TestClient(server.app), ssd
    storage.use(before)


def test_the_original_is_sent_unchanged_and_cached(media):
    client, ssd = media
    storage.use(storage.LocalStorage(ssd))
    r = client.get(f"/media/{media_token(RENDER)}")
    assert r.status_code == 200 and r.content == (ssd / RENDER).read_bytes()
    assert r.headers["cache-control"] == "private, max-age=86400" and "x-original" not in r.headers
    again = client.get(f"/media/{media_token(RENDER)}", headers={"If-None-Match": r.headers["etag"]})
    assert again.status_code == 304 and not again.content   # unchanged: not sent again


def test_without_the_drive_the_preview_stands_in_and_is_not_cached(media, tmp_path):
    client, ssd = media
    storage.use(storage.LocalStorage(tmp_path / "unplugged"))
    r = client.get(f"/media/{media_token(RENDER)}")
    assert r.status_code == 200 and r.content == b"RIFF small preview"
    assert r.headers["cache-control"] == "no-store" and r.headers["x-original"] == "unavailable"
    assert client.get(f"/media/{media_token(VIDEO)}").status_code == 503   # a video has no stand-in


def test_the_page_is_told_when_originals_are_offline(media, tmp_path, monkeypatch):
    from jewelsearch import auth
    client, ssd = media

    async def no(uid):
        return False
    monkeypatch.setattr(auth, "is_admin", no)
    monkeypatch.setattr(auth, "is_jeweler", no)
    storage.use(storage.LocalStorage(ssd))
    assert client.get("/api/auth/me").json()["originals"] is True
    storage.use(storage.LocalStorage(tmp_path / "unplugged"))
    assert client.get("/api/auth/me").json()["originals"] is False
