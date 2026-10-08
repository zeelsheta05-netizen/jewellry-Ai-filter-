"""Admin uploads, run against a temporary folder standing in for the dataset
drive and against a simulated S3 bucket (moto): the same behaviour from both."""
import asyncio

import pytest

from jewelsearch import storage, uploads

OWNER = "0f8fad5b-d9cb-469f-a165-70867728950e"


@pytest.fixture(params=["folder", "s3"])
def backend(request, tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "LOG", tmp_path / "uploads.log")
    monkeypatch.setattr(uploads, "RESERVE", 0)
    uploads._active.clear()
    if request.param == "folder":
        root = tmp_path / "Storage"
        root.mkdir()
        storage.use(storage.LocalStorage(root))
        yield storage.get()
    else:
        from moto import mock_aws
        for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "ap-south-1"}.items():
            monkeypatch.setenv(k, v)
        with mock_aws():
            import boto3
            boto3.client("s3", region_name="ap-south-1").create_bucket(
                Bucket="jewel-test", CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
            storage.use(storage.S3Storage("jewel-test", "ap-south-1", manifest=storage.Manifest(tmp_path / "m.sqlite"),
                                          cache_dir=tmp_path / "cache"))
            yield storage.get()
    storage.use(None)


@pytest.fixture
def drive(tmp_path):
    """Folder-only tests (empty folders, unplugging the drive)."""
    root = tmp_path / "Storage"
    root.mkdir()
    storage.use(storage.LocalStorage(root))
    uploads._active.clear()
    yield root
    storage.use(None)


def _read(path: str) -> bytes:
    return storage.get().read_bytes(path)


async def _body(*chunks):
    for c in chunks:
        yield c


def _send(up, data: bytes, chunk: int):
    async def go():
        res = None
        for off in range(0, max(len(data), 1), chunk):
            res = await uploads.append(up.id, OWNER, off, _body(data[off:off + chunk]))
        return res
    return asyncio.run(go())


def test_upload_lands_in_category_folder_with_original_name(backend):
    data = b"x" * 1000
    up = uploads.start(OWNER, "a@b.co", "bracelet", "DDBR-101@Y-#viwe1.png", len(data), batch="Loat - 15")
    res = _send(up, data, 300)
    assert res["done"] and res["path"] == "New Dataset/Bracelets/Loat - 15/DDBR-101@Y-#viwe1.png"
    assert _read(res["path"]) == data
    if isinstance(backend, storage.LocalStorage):
        assert not list((backend.root / "New Dataset/.incoming").glob("*.part"))
    assert uploads.recent()[0]["path"] == res["path"]


def test_large_upload_goes_in_parts(backend, monkeypatch):
    """Bigger than one S3 part: sent as a multipart upload, one 4 MB chunk at a time."""
    monkeypatch.setattr(storage, "PART_SIZE", 5 * 1024 * 1024)
    data = bytes(range(256)) * (13 * 1024 * 1024 // 256 + 7)
    up = uploads.start(OWNER, "a@b.co", "ring", "big.stl", len(data))
    res = _send(up, data, 4 * 1024 * 1024)
    assert res["done"] and _read(res["path"]) == data
    if isinstance(backend, storage.S3Storage):
        assert len(up.handle.parts) == 2 and backend.manifest.file(res["path"])["size"] == len(data)
        assert backend.manifest.file(res["path"])["key"] == "uploads/ring/big.stl"   # clean name in uploads/


def test_existing_file_is_never_overwritten(backend):
    for body in (b"first", b"second"):
        up = uploads.start(OWNER, "a@b.co", "ring", "R1.stl", len(body))
        res = _send(up, body, 100)
    assert res["path"] == "New Dataset/Rings/R1 (2).stl" and res["renamed"]
    assert _read("New Dataset/Rings/R1.stl") == b"first"


def test_two_uploads_of_one_name_get_different_names(backend):
    a = uploads.start(OWNER, "a@b.co", "ring", "R2.stl", 1)
    b = uploads.start(OWNER, "a@b.co", "ring", "R2.stl", 1)
    assert a.handle.path != b.handle.path
    assert {_send(a, b"a", 1)["path"], _send(b, b"b", 1)["path"]} == {"New Dataset/Rings/R2.stl", "New Dataset/Rings/R2 (2).stl"}


def test_failed_last_chunk_can_be_sent_again(backend, monkeypatch):
    up = uploads.start(OWNER, "a@b.co", "ring", "R3.stl", 6)
    _send_part = lambda off, data: asyncio.run(uploads.append(up.id, OWNER, off, _body(data)))  # noqa: E731
    assert _send_part(0, b"abc")["received"] == 3
    real = type(backend).commit_upload

    def broken(self, h):
        raise storage.StorageUnavailable("offline")
    monkeypatch.setattr(type(backend), "commit_upload", broken)
    with pytest.raises(uploads.UploadError) as e:
        _send_part(3, b"def")
    assert e.value.status == 503
    monkeypatch.setattr(type(backend), "commit_upload", real)
    assert _send_part(3, b"def")["done"]
    assert _read("New Dataset/Rings/R3.stl") == b"abcdef"


def test_folder_structure_kept_and_names_sanitised(backend):
    up = uploads.start(OWNER, "a@b.co", "necklace", 'a:b?.png', 3, subdirs="Set A/views")
    res = _send(up, b"abc", 10)
    assert res["path"] == "New Dataset/Necklaces/Set A/views/a_b_.png"


@pytest.mark.parametrize("name,subdirs", [
    (".DS_Store", ""), ("._x.png", ""), ("Thumbs.db", ""), ("..", ""),
    ("ok.png", "../../etc"), ("ok.png", "a/.hidden"),
])
def test_rejects_junk_and_path_escapes(name, subdirs, drive):
    with pytest.raises(uploads.UploadError):
        uploads.start(OWNER, "a@b.co", "ring", name, 1, subdirs=subdirs)


def test_rejects_unknown_category(drive):
    with pytest.raises(uploads.UploadError):
        uploads.start(OWNER, "a@b.co", "shoes", "x.png", 1)


def test_wrong_offset_and_other_user_rejected(backend):
    up = uploads.start(OWNER, "a@b.co", "ring", "x.png", 10)
    with pytest.raises(uploads.UploadError) as e:
        asyncio.run(uploads.append(up.id, OWNER, 5, _body(b"12345")))
    assert e.value.status == 409
    with pytest.raises(uploads.UploadError) as e:
        asyncio.run(uploads.append(up.id, "someone-else", 0, _body(b"12345")))
    assert e.value.status == 404


def test_oversized_chunk_rejected_and_can_be_resent(backend):
    up = uploads.start(OWNER, "a@b.co", "ring", "x.bin", 4)
    with pytest.raises(uploads.UploadError):
        asyncio.run(uploads.append(up.id, OWNER, 0, _body(b"12345")))
    res = _send(up, b"1234", 4)
    assert _read(res["path"]) == b"1234"


def test_empty_file_and_cancel(backend):
    res = _send(uploads.start(OWNER, "a@b.co", "other", "empty.txt", 0, custom="Misc"), b"", 1)
    assert res["done"] and _read(res["path"]) == b""
    up = uploads.start(OWNER, "a@b.co", "ring", "y.png", 10)
    uploads.cancel(up.id, OWNER)
    assert up.id not in uploads._active and not backend.exists(up.handle.path)
    if up.handle.part:
        assert not up.handle.part.exists()


def test_no_space_and_missing_drive(drive, monkeypatch):
    monkeypatch.setattr(uploads, "RESERVE", 10 ** 18)
    with pytest.raises(uploads.UploadError) as e:
        uploads.start(OWNER, "a@b.co", "ring", "x.png", 1)
    assert e.value.status == 507
    storage.use(storage.LocalStorage(drive / "unplugged"))
    with pytest.raises(uploads.UploadError) as e:
        uploads.start(OWNER, "a@b.co", "ring", "x.png", 1)
    assert e.value.status == 503


def test_s3_has_no_space_limit(backend):
    if isinstance(backend, storage.S3Storage):
        assert uploads.free_bytes() is None and uploads.new_root() == "s3://jewel-test/New Dataset"


@pytest.mark.parametrize("batch,expected", [
    ("AA", "Pendants/AA"),
    ("{root}/New Dataset/Pendants/AA", "Pendants/AA"),     # full path pasted in
    ("New Dataset/Pendants/BB", "Pendants/BB"),
    ("pendants/CC/Set 1", "Pendants/CC/Set 1"),
    ("{root}/New Dataset/Pendants", "Pendants"),
    ("  Loat - 15 ", "Pendants/Loat - 15"),
])
def test_typed_path_keeps_only_custom_folder_name(drive, batch, expected):
    d = uploads.target_dir("pendant", batch.format(root=drive))
    assert d == "New Dataset/" + expected


def test_pasted_bucket_path_keeps_only_custom_folder_name(backend):
    if isinstance(backend, storage.S3Storage):
        assert uploads.target_dir("pendant", "s3://jewel-test/New Dataset/Pendants/AA") == "New Dataset/Pendants/AA"


def test_path_to_other_category_rejected(drive):
    with pytest.raises(uploads.UploadError):
        uploads.target_dir("pendant", f"{drive}/New Dataset/Rings/AA")


def test_other_uses_admin_category_name(backend):
    up = uploads.start(OWNER, "a@b.co", "other", "M1.png", 2, batch="Set A", custom="Mangalsutra")
    assert _send(up, b"ok", 2)["path"] == "New Dataset/Mangalsutra/Set A/M1.png"
    assert uploads.custom_categories() == ["Mangalsutra"]
    # same name in other case / pasted as a path -> same folder; built-in names map to their folder
    name = lambda d: d.rsplit("/", 1)[-1]  # noqa: E731
    assert name(uploads.target_dir("other", custom="mangalsutra")) == "Mangalsutra"
    assert name(uploads.target_dir("other", custom=f"{backend.root_label()}/New Dataset/Nose Pins")) == "Nose Pins"
    assert name(uploads.target_dir("other", custom="rings")) == "Rings"
    assert uploads.target_dir("other", "Mangalsutra/AA", custom="Mangalsutra") == "New Dataset/Mangalsutra/AA"


@pytest.mark.parametrize("custom", ["", "   ", ".hidden", "~$x"])
def test_other_needs_a_valid_category_name(custom, drive):
    with pytest.raises(uploads.UploadError):
        uploads.target_dir("other", custom=custom)


def test_existing_folder_in_bucket_reported(backend):
    if isinstance(backend, storage.S3Storage):
        for name in ("a.png", "b.png"):
            _send(uploads.start(OWNER, "a@b.co", "pendant", name, 1, batch="AA"), b"x", 1)
        res = uploads.existing_folders("pendant", "", ["AA", "BB"])
        assert [(r["rel"], r["files"], r["suggestion"]) for r in res] == [("AA", 2, "AA (2)")]


def test_existing_folder_reported_with_free_name(drive):
    for name in ("a.png", "b.png"):
        _send(uploads.start(OWNER, "a@b.co", "pendant", name, 1, batch="AA"), b"x", 1)
    (drive / "New Dataset/Pendants/AA (2)").mkdir()
    (drive / "New Dataset/Pendants/Loat - 1.5").mkdir()
    res = uploads.existing_folders("pendant", "", ["AA", "BB", "Loat - 1.5", ""])
    assert [(r["rel"], r["files"], r["suggestion"]) for r in res] == [
        ("AA", 2, "AA (3)"), ("Loat - 1.5", 0, "Loat - 1.5 (2)")]
