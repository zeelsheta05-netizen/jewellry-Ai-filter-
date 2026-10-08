"""The dataset storage and the move to S3, end to end on a tiny fake SSD and a
simulated S3 bucket (moto): what lands where, compression, de-duplication,
reading back by name, signed links, verification and recovery of the index."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from jewelsearch import storage

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "jewel-test"
REGION = "ap-south-1"
DESIGN = "01/Loat - 1/Ring/DDLR-1"
RENDER = f"{DESIGN}/DDLR-1@W-#viwe1.png"
COPY = "PHOTOS/DDLR-1@W-#viwe1.png"            # same bytes in another folder
VIDEO = f"{DESIGN}/DDLR-1-02@W-#viwe5.mp4"
STL = f"{DESIGN}/DDLR-1.stl"
CAD = f"{DESIGN}/DDLR-1.3dm"
CARD = "01/Loat - 1/xlsx/DDLR-1.xlsx"


def _load_migrate():
    spec = importlib.util.spec_from_file_location("migrate_to_s3", ROOT / "scripts" / "migrate_to_s3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png() -> bytes:
    im = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    for x in range(20, 76):
        for y in range(20, 76):
            im.putpixel((x, y), (200, 160 + x % 40, 90, 255))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def ssd(tmp_path):
    root = tmp_path / "Storage"
    files = {
        RENDER: _png(), COPY: _png(), VIDEO: os.urandom(30_000),
        STL: b"solid ring\n" + b"facet normal 0 0 1\n outer loop\n  vertex 1 2 3\n endloop\nendfacet\n" * 400,
        CAD: os.urandom(9_000), CARD: b"PK fake xlsx",
        "01/._DDLR-1.png": b"junk", "01/Thumbs.db": b"junk",
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


@pytest.fixture
def env(tmp_path, monkeypatch, ssd):
    from moto import mock_aws
    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": REGION,
                 "S3_REGION": REGION, "S3_BUCKET": BUCKET}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    monkeypatch.setattr(storage, "MANIFEST", tmp_path / "state" / "manifest.sqlite")
    monkeypatch.setattr(storage, "CACHE", tmp_path / "state" / "cache")
    monkeypatch.setattr(storage, "TMP", tmp_path / "state" / "tmp")
    mig = _load_migrate()
    index = tmp_path / "index"
    index.mkdir()
    meta = {"design_id": "DDLR-1", "folders": [DESIGN], "key": f"{DESIGN}/DDLR-1", "category": "ring",
            "images": {"white_gold": {"1": RENDER}}, "videos": {"white_gold": VIDEO}}
    (index / "meta.jsonl").write_text(json.dumps(meta) + "\n")
    monkeypatch.setattr(mig, "INDEX", index)
    monkeypatch.setattr(mig, "DATA", tmp_path / "data")
    env_file = tmp_path / ".env"
    env_file.write_text(f"# test settings\nJEWEL_STORAGE={ssd.as_uri()}\nS3_BUCKET={BUCKET}\n")
    monkeypatch.setattr(mig, "ENV", env_file)
    monkeypatch.setenv("JEWEL_STORAGE", ssd.as_uri())
    monkeypatch.delenv("JEWEL_STORAGE_FALLBACK", raising=False)
    monkeypatch.setattr(mig, "bucket_is_public", lambda s3, key: False)   # no real internet in tests
    monkeypatch.setattr(mig, "signed_link_works", lambda s3, key, body: True)
    with mock_aws():
        import boto3
        boto3.client("s3", region_name=REGION).create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        yield mig, ssd
    storage.use(None)


def _args(**kw):
    base = {"bucket": "", "source": "", "estimate": False, "versioning": False, "force": False, "videos": "original", "workers": 1, "video_workers": 1,
            "limit": 0, "sample": 50, "no_search_backup": True}
    return argparse.Namespace(**{**base, **kw})


def _migrate(mig, ssd):
    mig.cmd_check(_args())
    mig.cmd_setup(_args())
    mig.cmd_scan(_args(source=str(ssd)))
    mig.cmd_web(_args(source=str(ssd)))
    mig.cmd_originals(_args(source=str(ssd)))
    mig.cmd_verify(_args())


def test_move_compress_dedupe_and_read_back(env):
    mig, ssd = env
    _migrate(mig, ssd)
    s3 = storage.S3Storage(BUCKET, REGION)
    man = s3.manifest
    keys = {o["Key"]: o for o in s3.client.list_objects_v2(Bucket=BUCKET)["Contents"]}

    # every data file once; junk never moved
    assert man.db.execute("SELECT COUNT(*) FROM scan").fetchone()[0] == 6
    assert not any("Thumbs" in k or "._" in k for k in keys)
    # the PHOTOS copy points at the first copy's object: stored once
    assert man.file(COPY)["key"] == man.file(RENDER)["key"]
    assert sum(1 for k in keys if k.endswith(".png")) == 1
    # clean, readable names: the catalogue by category/design/metal, the design's other files beside it
    assert man.file(RENDER)["key"] == "catalogue/ring/ddlr-1/white-gold/view-1.png"
    assert man.file(VIDEO)["key"] == "catalogue/ring/ddlr-1/white-gold/video.mp4"
    assert man.file(CARD)["key"] == "library/mixed/lot-1/xlsx/ddlr-1.xlsx"
    meta = s3.client.head_object(Bucket=BUCKET, Key=man.file(RENDER)["key"])["Metadata"]
    assert "viwe1" in meta["source"]                      # the dataset name travels with the object
    # CAD text compresses: kept as .zst; random bytes and xlsx stay as they are
    assert man.file(STL)["codec"] == "zstd" and man.file(STL)["key"] == "catalogue/ring/ddlr-1/files/ddlr-1.stl.zst"
    assert man.file(STL)["stored"] < (ssd / STL).stat().st_size / 5
    assert man.file(CAD)["codec"] == "" and man.file(CARD)["codec"] == ""
    # catalogue pictures are served as the originals: no WebP copy is made unless asked for
    web = storage.web_key("ring", storage.design_slug("DDLR-1", DESIGN), "white_gold", 1)
    assert web not in keys and man.web_copy(RENDER) is None
    assert man.design("DDLR-1")[0]["path"] in (RENDER, VIDEO)
    assert keys["catalogue/ring/ddlr-1/files/ddlr-1.3dm"]["StorageClass"] == "INTELLIGENT_TIERING"

    # reading back by name gives the SSD's bytes exactly
    for rel in (RENDER, COPY, STL, CAD, CARD, VIDEO):
        assert s3.read_bytes(rel) == (ssd / rel).read_bytes(), rel
    assert s3.read_bytes(RENDER, prefer="web") == (ssd / RENDER).read_bytes()   # no copy: the original
    local = s3.local_copy(STL)
    assert local == s3.cache_dir / STL and local.read_bytes() == (ssd / STL).read_bytes()

    # browser links: signed, to the original as it is, the same link reused (browser cache)
    url = s3.media_url(RENDER)
    assert url.startswith(f"https://{BUCKET}.s3.{REGION}.amazonaws.com/catalogue/ring/ddlr-1/") and "X-Amz-Signature" in url
    assert s3.media_url(RENDER) == url
    assert "/white-gold/video.mp4" in s3.media_url(VIDEO)
    with pytest.raises(FileNotFoundError):
        s3.media_url(STL)            # stored compressed: never handed to a browser
    with pytest.raises(FileNotFoundError):
        s3.read_bytes("01/nothing.png")


def test_rerun_skips_finished_work(env):
    mig, ssd = env
    _migrate(mig, ssd)
    man = storage.Manifest()
    before = man.db.execute("SELECT key, uploaded_at FROM objects ORDER BY key").fetchall()
    mig.cmd_web(_args(source=str(ssd)))
    mig.cmd_originals(_args(source=str(ssd)))
    after = man.db.execute("SELECT key, uploaded_at FROM objects ORDER BY key").fetchall()
    assert [tuple(r) for r in before] == [tuple(r) for r in after]


def test_verify_catches_a_missing_object(env):
    mig, ssd = env
    _migrate(mig, ssd)
    s3 = storage.S3Storage(BUCKET, REGION)
    s3.client.delete_object(Bucket=BUCKET, Key="catalogue/ring/ddlr-1/files/ddlr-1.3dm")
    with pytest.raises(SystemExit):
        mig.cmd_verify(_args())
    mig.cmd_originals(_args(source=str(ssd)))      # sends the missing file again
    mig.cmd_verify(_args())
    assert s3.read_bytes(CAD) == (ssd / CAD).read_bytes()


def test_new_server_gets_the_index_from_the_bucket(env):
    mig, ssd = env
    _migrate(mig, ssd)
    mig.cmd_index(_args())
    storage.Manifest().close()
    for f in storage.MANIFEST.parent.glob("manifest.sqlite*"):
        f.unlink()
    s3 = storage.S3Storage(BUCKET, REGION)
    assert s3.ready()
    assert s3.read_bytes(CAD) == (ssd / CAD).read_bytes()


def test_listing_comes_from_the_index(env):
    mig, ssd = env
    _migrate(mig, ssd)
    s3 = storage.S3Storage(BUCKET, REGION)
    assert sorted(s3.walk("01/Loat - 1")) == sorted([RENDER, VIDEO, STL, CAD, CARD])
    assert list(s3.walk("01", exts={".3dm"})) == [CAD]
    assert s3.subdirs("01/Loat - 1") == ["Ring", "xlsx"]
    assert s3.exists(RENDER) and not s3.exists(f"{DESIGN}/other.png")
    # the dataset audit runs on the bucket's index exactly as it did on the SSD
    spec = importlib.util.spec_from_file_location("audit_dataset", ROOT / "scripts" / "audit_dataset.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    designs, _ = audit.audit(s3)
    on_ssd, _ = audit.audit(storage.LocalStorage(ssd))
    assert designs.keys() == on_ssd.keys()
    rec = designs[(DESIGN, "DDLR-1")]
    assert rec["images"] == {"white_gold": {1: RENDER}} and rec["has_cad"]


def test_media_route_redirects_to_signed_link(env, monkeypatch):
    mig, ssd = env
    _migrate(mig, ssd)
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    from jewelsearch.config import media_token
    storage.use(storage.S3Storage(BUCKET, REGION))
    monkeypatch.setattr(server, "engine", type("E", (), {"media_by_token": {media_token(p): p for p in (RENDER, VIDEO)}})())
    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u1", "email": "a@b.co", "name": "A"})

    async def ok(uid):
        return True
    monkeypatch.setattr(auth, "is_approved", ok)
    client = TestClient(server.app)
    r = client.get(f"/media/{media_token(RENDER)}", follow_redirects=False)
    assert r.status_code == 302 and "/catalogue/" in r.headers["location"]   # the render as it is
    assert client.get(f"/media/{media_token(STL)}", follow_redirects=False).status_code == 404   # not a catalogue file
    # pages can't name a file by path any more, and IDs must look like IDs
    assert client.get("/media", params={"path": RENDER}, follow_redirects=False).status_code == 404
    assert client.get("/media/..%2F..%2Fetc", follow_redirects=False).status_code in (404, 422)
    # signed out: nothing
    monkeypatch.setattr(auth, "read_session", lambda c: None)
    assert client.get(f"/media/{media_token(RENDER)}", follow_redirects=False).status_code == 401


def test_csp_allows_the_bucket_for_media(env):
    s3 = storage.S3Storage(BUCKET, REGION)
    assert s3.media_origin() == f"https://{BUCKET}.s3.{REGION}.amazonaws.com"


def test_dataset_names_cannot_escape(tmp_path):
    st = storage.LocalStorage(tmp_path)
    for bad in ("../etc/passwd", "/etc/passwd", "a/../../b", "a//b", ""):
        with pytest.raises(FileNotFoundError):
            st.read_bytes(bad)


def test_storage_url_setting():
    assert isinstance(storage.from_url("s3://my-bucket"), storage.S3Storage)
    assert storage.from_url("s3://my-bucket").bucket == "my-bucket"
    local = storage.from_url("file:///Volumes/Storage")
    assert isinstance(local, storage.LocalStorage) and str(local.root) == "/Volumes/Storage"
    with pytest.raises(ValueError):
        storage.from_url("/Volumes/Storage")


def test_cache_keeps_to_its_size(tmp_path):
    s3 = storage.S3Storage("b", REGION, manifest=storage.Manifest(tmp_path / "m.sqlite"), cache_dir=tmp_path / "c",
                           cache_bytes=100)
    old = tmp_path / "c" / "a" / "old.bin"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"x" * 80)
    os.utime(old, (1, 1))
    new = tmp_path / "c" / "a" / "new.bin"
    new.write_bytes(b"y" * 80)
    s3._evict(keep=new)
    assert new.exists() and not old.exists()


def test_sha_of_upload_matches(env):
    """What an admin upload records is the file's real SHA-256 (verify relies on it)."""
    mig, ssd = env
    s3 = storage.S3Storage(BUCKET, REGION)
    assert s3.ready()
    h = s3.begin_upload("New Dataset/Rings/x.stl", 5)
    s3.write_upload(h, b"hello")
    s3.commit_upload(h)
    assert s3.manifest.file("New Dataset/Rings/x.stl")["sha256"] == hashlib.sha256(b"hello").hexdigest()


def test_video_web_copy_is_smaller_and_served(env):
    """--videos encode: a smaller H.264 copy for the browser, the original kept as it is."""
    import subprocess
    mig, ssd = env
    exe = mig.ffmpeg_exe()
    if not exe:
        pytest.skip("no ffmpeg")
    # a 2 s lossless test video: big, like the studio turntables
    subprocess.run([exe, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
                    "-t", "2", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(ssd / VIDEO)], check=True)
    mig.cmd_scan(_args(source=str(ssd)))
    mig.cmd_web(_args(source=str(ssd), videos="encode"))
    s3 = storage.S3Storage(BUCKET, REGION)
    web = s3.manifest.web_copy(VIDEO)
    assert web["kind"] == "video" and web["key"].endswith("/white_gold/turntable.mp4")
    small = s3.client.head_object(Bucket=BUCKET, Key=web["key"])["ContentLength"]
    assert small < 0.8 * (ssd / VIDEO).stat().st_size
    assert "turntable.mp4" in s3.media_url(VIDEO)
    assert s3.read_bytes(VIDEO) == (ssd / VIDEO).read_bytes()       # the original is untouched


def test_pictures_are_served_as_originals_even_with_web_copies(env):
    """--image-web-copies makes WebP copies, but a catalogue picture is still sent as the original."""
    mig, ssd = env
    mig.cmd_check(_args())
    mig.cmd_setup(_args())
    mig.cmd_scan(_args(source=str(ssd)))
    mig.cmd_web(_args(source=str(ssd), image_web_copies=True))
    mig.cmd_originals(_args(source=str(ssd)))
    s3 = storage.S3Storage(BUCKET, REGION)
    assert s3.manifest.web_copy(RENDER) is not None                   # the copy exists...
    assert "/catalogue/" in s3.media_url(RENDER)                      # ...but is never shown
    assert s3.read_bytes(RENDER) == (ssd / RENDER).read_bytes()


def test_s3_compatible_services_get_checksums_only_when_required(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    r2 = storage.S3Storage("b", "auto", "https://acc.r2.cloudflarestorage.com", manifest=storage.Manifest("/dev/null/x"))
    assert r2.client.meta.config.request_checksum_calculation == "when_required"
    assert r2.media_origin() == "https://acc.r2.cloudflarestorage.com"
    assert r2.originals_class == "" and r2._conditional() == {}
    aws = storage.S3Storage("b", REGION, manifest=storage.Manifest("/dev/null/x"))
    assert aws.client.meta.config.request_checksum_calculation != "when_required"


def test_setup_can_turn_on_versioning(env):
    mig, _ = env
    mig.cmd_setup(_args(versioning=True))
    s3 = storage.S3Storage(BUCKET, REGION)
    assert s3.client.get_bucket_versioning(Bucket=BUCKET)["Status"] == "Enabled"


# ---------------------------------------------------------------- the switch-over


def test_connect_switches_the_app_with_the_ssd_as_fallback(env):
    mig, ssd = env
    mig.cmd_connect(_args())
    text = mig.ENV.read_text()
    assert f"JEWEL_STORAGE=s3://{BUCKET}" in text and f"JEWEL_STORAGE_FALLBACK={ssd.as_uri()}" in text
    assert "# test settings" in text and oct(mig.ENV.stat().st_mode)[-3:] == "600"
    st = storage.configured(f"s3://{BUCKET}", ssd.as_uri())
    assert isinstance(st, storage.FallbackStorage) and st.ready()
    # nothing moved yet: everything still comes from the SSD
    assert st.read_bytes(CAD) == (ssd / CAD).read_bytes()
    assert st.media_url(RENDER) is None and st.local_copy(RENDER) == ssd / RENDER


def test_transition_reads_bucket_first_and_uploads_only_to_bucket(env):
    mig, ssd = env
    _migrate(mig, ssd)
    st = storage.configured(f"s3://{BUCKET}", ssd.as_uri())
    (ssd / "01/only-on-ssd.stl").write_bytes(b"not moved yet")
    assert "/catalogue/" in st.media_url(RENDER)                      # moved: from the bucket
    assert st.read_bytes("01/only-on-ssd.stl") == b"not moved yet"    # not moved: from the SSD
    assert "01/only-on-ssd.stl" in list(st.walk("01"))
    with pytest.raises(FileExistsError):                              # never hide an SSD file behind a new upload
        st.begin_upload("01/only-on-ssd.stl", 3)
    h = st.begin_upload("New Dataset/Rings/new.stl", 3)
    st.write_upload(h, b"new")
    st.commit_upload(h)
    assert not (ssd / "New Dataset/Rings/new.stl").exists()           # the SSD is never written
    assert st.primary.read_bytes("New Dataset/Rings/new.stl") == b"new"


def test_move_then_finish_stops_using_the_ssd(env):
    mig, ssd = env
    mig.cmd_connect(_args())
    with pytest.raises(SystemExit):
        mig.cmd_finish(_args())                                       # nothing moved yet: refused
    assert "JEWEL_STORAGE_FALLBACK" in mig.ENV.read_text()
    mig.cmd_move(_args())
    mig.cmd_finish(_args())
    text = mig.ENV.read_text()
    assert "JEWEL_STORAGE_FALLBACK" not in text and f"JEWEL_STORAGE=s3://{BUCKET}" in text
    st = storage.configured(f"s3://{BUCKET}")
    for rel in (RENDER, COPY, VIDEO, STL, CAD, CARD):
        assert st.read_bytes(rel) == (ssd / rel).read_bytes(), rel


def test_restore_brings_back_the_indexes(env, tmp_path):
    mig, ssd = env
    data = tmp_path / "data"
    (data / "index").mkdir(parents=True)
    (data / "index" / "meta.jsonl").write_text("{}\n")
    (data / "crops").mkdir()
    (data / "crops" / "a.webp").write_bytes(b"crop")
    (data / "bodyphotos").mkdir()
    (data / "bodyphotos" / "face.jpg").write_bytes(b"personal")
    _migrate(mig, ssd)
    mig.cmd_index(_args(no_search_backup=False))
    import shutil
    shutil.rmtree(data)
    storage.Manifest().close()
    for f in storage.MANIFEST.parent.glob("manifest.sqlite*"):
        f.unlink()
    mig.INDEX = data / "index"
    mig.cmd_restore(_args(force=False))
    assert (data / "crops" / "a.webp").read_bytes() == b"crop" and (data / "index" / "meta.jsonl").exists()
    assert not (data / "bodyphotos").exists()                        # personal photos never went to the bucket
    assert storage.Manifest().file(CAD) is not None


def test_check_refuses_a_public_bucket(env, monkeypatch):
    mig, _ = env
    monkeypatch.setattr(mig, "bucket_is_public", lambda s3, key: True)
    with pytest.raises(SystemExit, match="public"):
        mig.cmd_check(_args())


def test_missing_setting_is_reported_not_guessed(monkeypatch):
    st = storage.configured("")
    assert isinstance(st, storage.UnconfiguredStorage) and not st.ready()
    with pytest.raises(storage.StorageUnavailable, match="JEWEL_STORAGE"):
        st.read_bytes(RENDER)
    with pytest.raises(storage.StorageUnavailable):
        st.media_url(RENDER)


def test_admin_panel_sees_cloud_storage_during_the_move(tmp_path):
    st = storage.configured("s3://b", tmp_path.as_uri())
    assert storage.kind(st) == "s3" and storage.kind(storage.configured(tmp_path.as_uri())) == "folder"
    assert storage.kind(storage.configured("")) == "none"


def test_check_refuses_when_signed_links_fail(env, monkeypatch):
    mig, _ = env
    monkeypatch.setattr(mig, "signed_link_works", lambda s3, key, body: False)
    with pytest.raises(SystemExit, match="signed link"):
        mig.cmd_check(_args())


def test_browser_style_requests_fail_loudly_when_unreachable():
    """A check that can't connect must not report "private" or "works"."""
    mig = _load_migrate()
    with pytest.raises(Exception):
        mig._https_get("https://127.0.0.1:9/never")


def test_check_leaves_nothing_behind_in_a_versioned_bucket(env):
    """With versioning on (Zata, DigitalOcean), a plain delete keeps the old copy as a
    hidden version that still uses paid space: the check removes every version."""
    mig, _ = env
    mig.cmd_setup(_args(versioning=True))
    mig.cmd_check(_args())
    mig.cmd_check(_args())
    c = storage.S3Storage(BUCKET, REGION).client
    r = c.list_object_versions(Bucket=BUCKET, Prefix=storage.INDEX + ".")
    assert not r.get("Versions") and not r.get("DeleteMarkers")


def test_check_reports_whether_overwrites_are_refused(env, capsys):
    mig, _ = env
    mig.cmd_check(_args())
    out = capsys.readouterr().out
    assert "large files (multipart)  work" in out
    assert "S3_CONDITIONAL_WRITES" in out


def test_conditional_write_probe_reads_the_providers_answer():
    from botocore.exceptions import ClientError
    mig = _load_migrate()

    class Refuses:
        def put_object(self, **kw):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")

    class Ignores:
        def put_object(self, **kw):
            return {}

    class Rejects:
        def put_object(self, **kw):
            raise ClientError({"Error": {"Code": "NotImplemented"}}, "PutObject")

    assert mig.conditional_writes_work(Refuses(), "b", "k", b"x")
    assert not mig.conditional_writes_work(Ignores(), "b", "k", b"x")
    assert not mig.conditional_writes_work(Rejects(), "b", "k", b"x")


# ---------------------------------------------------------------- partial moves (--max-gb)


def test_partial_move_sends_catalogue_pictures_first_and_stops_at_the_limit(env):
    mig, ssd = env
    mig.cmd_scan(_args(source=str(ssd)))
    render = (ssd / RENDER).stat().st_size
    mig.cmd_originals(_args(source=str(ssd), max_gb=(render + 10) / 1e9))
    man = storage.Manifest()
    assert man.file(RENDER) is not None and man.file(COPY) is not None   # the picture, and its copy for free
    assert man.file(VIDEO) is None and man.file(CAD) is None and man.file(STL) is None
    mig.cmd_originals(_args(source=str(ssd), max_gb=1.0))                # a bigger limit continues
    assert all(man.file(p) is not None for p in (VIDEO, CAD, STL, CARD))


def test_partial_move_is_not_reported_as_complete(env, capsys):
    mig, ssd = env
    render = (ssd / RENDER).stat().st_size
    mig.cmd_move(_args(source=str(ssd), max_gb=(render + 10) / 1e9))
    assert "Part of the dataset is in the bucket" in capsys.readouterr().out
    man = storage.Manifest()
    assert man.file(RENDER) is not None and man.file(VIDEO) is None     # the video waited: no room left
    with pytest.raises(SystemExit, match="not in the bucket yet"):
        mig.cmd_finish(_args())
    mig.cmd_move(_args(source=str(ssd)))                                 # without a limit: the rest
    assert "complete and verified" in capsys.readouterr().out
    mig.cmd_finish(_args())


def test_a_full_bucket_stops_the_move_cleanly(env, monkeypatch):
    from botocore.exceptions import ClientError
    mig, ssd = env
    mig.cmd_scan(_args(source=str(ssd)))
    real, tried = mig.put_original, []

    def put(s3, src, rel, size, key):
        tried.append(rel)
        if len(tried) > 1:
            raise ClientError({"Error": {"Code": "QuotaExceeded", "Message": ""}}, "PutObject")
        return real(s3, src, rel, size, key)

    monkeypatch.setattr(mig, "put_original", put)
    with pytest.raises(SystemExit, match="QuotaExceeded"):
        mig.cmd_originals(_args(source=str(ssd)))
    assert storage.Manifest().file(RENDER) is not None    # what went before is recorded
    assert len(tried) == 2                                # the rest was not tried (5 files to send)


def test_the_limit_skips_what_does_not_fit_but_keeps_the_order():
    mig = _load_migrate()
    assert mig.within([5, 9, 3, 1], 9, lambda x: x) == ([5, 3, 1], 1)
    assert mig.within([5, 9], None, lambda x: x) == ([5, 9], 0)
    key = mig.move_priority({"a/r@W-#viwe1.png": 1, "a/r@W-#viwe5.mp4": 5})
    order = sorted(["a/r.3dm", "a/r@W-#viwe5.mp4", "b/photo.jpg", "a/r@W-#viwe1.png", "c/other.mp4"], key=key)
    assert order == ["a/r@W-#viwe1.png", "b/photo.jpg", "a/r@W-#viwe5.mp4", "c/other.mp4", "a/r.3dm"]


def test_category_move_sends_only_that_categorys_catalogue(env):
    mig, ssd = env
    mig.cmd_scan(_args(source=str(ssd)))
    with pytest.raises(SystemExit, match="Choose one of: ring"):       # a typo sends nothing
        mig.cmd_originals(_args(source=str(ssd), category="rings"))
    man = storage.Manifest()
    assert man.file(RENDER) is None
    mig.cmd_originals(_args(source=str(ssd), category="Ring"))
    assert man.file(RENDER) is not None and man.file(VIDEO) is not None
    assert man.file(CAD) is None and man.file(CARD) is None            # not catalogue files: they wait


def test_category_move_with_a_limit_sends_pictures_first(env, capsys):
    mig, ssd = env
    render = (ssd / RENDER).stat().st_size
    mig.cmd_move(_args(source=str(ssd), category="ring", max_gb=(render + 10) / 1e9))
    man = storage.Manifest()
    assert man.file(RENDER) is not None and man.file(VIDEO) is None
    assert "Part of the dataset is in the bucket" in capsys.readouterr().out


def test_a_wrong_bucket_name_stops_before_anything_is_sent(env, monkeypatch):
    mig, ssd = env
    monkeypatch.setenv("S3_BUCKET", "sub54a1-designfinder")       # the user name typed as the bucket
    with pytest.raises(SystemExit, match="check S3_BUCKET"):
        mig.cmd_move(_args(source=str(ssd)))
    assert not storage.Manifest().db.execute("SELECT 1 FROM scan LIMIT 1").fetchone()   # stopped before the scan


def test_transition_reads_the_ssd_when_the_bucket_does_not_answer(env, monkeypatch):
    mig, ssd = env
    _migrate(mig, ssd)
    s3 = storage.S3Storage(BUCKET, REGION)
    both = storage.FallbackStorage(s3, storage.LocalStorage(ssd))

    def down(*a, **kw):
        raise storage.StorageUnavailable("bucket did not answer")

    monkeypatch.setattr(s3, "read_bytes", down)
    assert both.read_bytes(RENDER) == (ssd / RENDER).read_bytes()


def test_clean_bucket_names():
    assert storage.clean_part(" Loat - 19 ") == "lot-19"
    assert storage.clean_part("Gent's Ring") == "gents-ring"
    assert storage.clean_name("DDLR-434@R-#viwe4.PNG") == "ddlr-434-r-view4.png"
    assert storage.library_key("02/02/Loat - 19 /rings/DDLR-852/DDLR-852/DDLR-852.3dm") == \
        "library/ring/lot-19/ddlr-852/ddlr-852.3dm"
    assert storage.library_key("EARRINGS NOT UPLOAD/9#10017E/9#10017E/9#10017E.stl") == \
        "library/earrings/not-uploaded/9-10017e/9-10017e.stl"
    assert storage.library_key("03/03/Loat - 26 /PHOTOS/DDLR-250 -AC-02@R-#viwe4.png") == \
        "library/mixed/lot-26/photos/ddlr-250-ac-02-r-view4.png"
    assert storage.library_key("New Dataset/Rings/Loat - 15/a b.png") == "uploads/ring/lot-15/a-b.png"
    assert storage.catalogue_key("ring", "ddlr-1", "rose_gold", 5, "x@R-#viwe5.MP4") == \
        "catalogue/ring/ddlr-1/rose-gold/video.mp4"


def test_plan_links_design_files_and_keeps_names_unique(tmp_path):
    mig = _load_migrate()
    rows = [("L/Ring/A1/A1", "A1", "A1-aaaaaa", "ring", "white_gold", 1, "L/Ring/A1/A1@W-#viwe1.png"),
            ("M/Ring/A1/A1", "A1", "A1-bbbbbb", "ring", "white_gold", 1, "M/Ring/A1/A1@W-#viwe1.png")]
    man = storage.Manifest(tmp_path / "manifest.sqlite")     # never the real storage index
    with man.write() as con:
        con.executemany("INSERT INTO scan VALUES (?,?,0,NULL)", [(p, 1) for p in (
            "L/Ring/A1/A1@W-#viwe1.png", "M/Ring/A1/A1@W-#viwe1.png",
            "L/Ring/A1/A1/A1.stl",          # deeper, with the id: the design's file
            "L/Ring/A1/B7/B7.stl",          # deeper, another design's folder: library
            "L/Ring/A1/a b.png", "L/Ring/A1/a-b.png")])   # two names that clean to the same name
    assert mig.plan_layout(man, rows) == 6
    key = dict(man.db.execute("SELECT path, key FROM layout").fetchall())
    # the same id twice in a category: the slug's hash keeps the folders apart
    assert key["L/Ring/A1/A1@W-#viwe1.png"] == "catalogue/ring/a1-aaaaaa/white-gold/view-1.png"
    assert key["M/Ring/A1/A1@W-#viwe1.png"] == "catalogue/ring/a1-bbbbbb/white-gold/view-1.png"
    assert key["L/Ring/A1/A1/A1.stl"] == "catalogue/ring/a1-aaaaaa/files/a1/a1.stl"
    assert key["L/Ring/A1/B7/B7.stl"].startswith("library/ring/")
    assert {key["L/Ring/A1/a b.png"], key["L/Ring/A1/a-b.png"]} == \
        {"catalogue/ring/a1-aaaaaa/files/a-b.png", "catalogue/ring/a1-aaaaaa/files/a-b-2.png"}
    assert mig.plan_layout(man, rows) == 0               # planned names never change


def test_relayout_renames_old_names_inside_the_bucket(env):
    mig, ssd = env
    _migrate(mig, ssd)
    s3 = storage.S3Storage(BUCKET, REGION)
    man = s3.manifest
    s3.client.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    # make it look like the old layout: objects named originals/<dataset name>, no plan yet
    with man.write() as con:
        for o in con.execute("SELECT key FROM objects WHERE tier='original'").fetchall():
            f = con.execute("SELECT path FROM files WHERE key = ? ORDER BY path LIMIT 1", (o[0],)).fetchone()[0]
            old = "originals/" + f + (".zst" if o[0].endswith(".zst") else "")
            s3.client.copy_object(Bucket=BUCKET, Key=old, CopySource={"Bucket": BUCKET, "Key": o[0]})
            mig.remove_every_version(s3.client, BUCKET, o[0])
            con.execute("UPDATE objects SET key = ? WHERE key = ?", (old, o[0]))
            con.execute("UPDATE files SET key = ? WHERE key = ?", (old, o[0]))
        con.execute("DELETE FROM layout")
    mig.cmd_relayout(_args(workers=2))
    assert man.file(RENDER)["key"] == "catalogue/ring/ddlr-1/white-gold/view-1.png"
    assert man.file(COPY)["key"] == man.file(RENDER)["key"]          # copies follow their object
    versions = s3.client.list_object_versions(Bucket=BUCKET, Prefix="originals/")
    assert not versions.get("Versions") and not versions.get("DeleteMarkers")   # old names gone for good
    for rel in (RENDER, COPY, STL, CAD, CARD, VIDEO):
        assert s3.read_bytes(rel) == (ssd / rel).read_bytes(), rel
    mig.cmd_verify(_args())
    mig.cmd_relayout(_args(workers=2))                  # nothing left to do
