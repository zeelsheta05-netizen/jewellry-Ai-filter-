"""Search by photo: reading uploads safely, finding the piece, and the
photo-search API (with a fake engine, so no model is loaded)."""
import base64
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from jewelsearch import photo


def jpeg(im: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92, **kw)
    return buf.getvalue()


def png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def data_url(b: bytes, kind="jpeg") -> str:
    return f"data:image/{kind};base64," + base64.b64encode(b).decode()


def piece_on(bg, colour, size=(800, 600), box=(300, 200, 500, 400)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.ellipse(box, fill=colour)
    d.ellipse((box[0] + 50, box[1] + 50, box[2] - 50, box[3] - 50), fill=bg)   # a ring: a hole in the middle
    return im


# ---- reading uploads -------------------------------------------------------

def test_data_url_must_be_a_base64_image():
    with pytest.raises(photo.PhotoError):
        photo.from_data_url("data:text/plain;base64,aGVsbG8=")
    with pytest.raises(photo.PhotoError):
        photo.from_data_url("data:image/jpeg;base64,***not base64***")
    assert photo.from_data_url(data_url(b"abc")) == b"abc"


def test_oversized_upload_is_refused(monkeypatch):
    monkeypatch.setattr(photo, "MAX_BYTES", 10)
    with pytest.raises(photo.PhotoError) as e:
        photo.from_data_url(data_url(b"x" * 11))
    assert e.value.status == 413


def test_not_an_image_is_refused():
    with pytest.raises(photo.PhotoError):
        photo.read(b"%PDF-1.4 not an image")


def test_tiny_photo_is_refused():
    with pytest.raises(photo.PhotoError):
        photo.read(jpeg(Image.new("RGB", (40, 40), "white")))


def test_too_many_pixels_is_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(photo, "MAX_PIXELS", 100 * 100)
    with pytest.raises(photo.PhotoError) as e:
        photo.read(png(Image.new("RGB", (200, 200), "white")))
    assert e.value.status == 413


def test_large_photo_is_shrunk_and_rgb():
    im = photo.read(jpeg(Image.new("RGB", (3000, 2000), "white")))
    assert im.mode == "RGB" and max(im.size) == photo.WORK_SIDE


def test_phone_orientation_is_applied():
    # a phone stores a portrait photo as landscape pixels + an EXIF rotation tag
    im = Image.new("RGB", (400, 200), "white")
    exif = Image.Exif()
    exif[0x0112] = 6   # rotate 90° clockwise to display
    out = photo.read(jpeg(im, exif=exif.tobytes()))
    assert out.size == (200, 400)


def test_transparent_cutout_goes_on_white():
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((50, 50, 150, 150), fill=(230, 180, 80, 255))
    out = photo.read(png(im))
    assert out.getpixel((5, 5)) == (255, 255, 255)


# ---- finding the piece -----------------------------------------------------

def test_piece_is_found_on_a_plain_background():
    im = piece_on((25, 30, 60), (225, 185, 90))
    v = photo.views(im)
    assert v["mask"] is not None and "piece" in v
    x0, y0, x1, y1 = v["box"]
    # the box holds the ring (x 300-500 of 800, y 200-400 of 600) with the catalogue's padding
    assert 0.3 < x0 < 0.375 and 0.625 < x1 < 0.7 and 0.27 < y0 < 0.34 and 0.66 < y1 < 0.73
    assert v["piece"].size[0] == v["piece"].size[1]


def test_piece_is_found_on_a_gradient_with_a_shadow():
    w, h = 800, 600
    a = np.zeros((h, w, 3), np.float32)
    a[:] = np.linspace(170, 230, w)[None, :, None] * np.array([1.0, 0.9, 0.8])
    bg = Image.fromarray(a.astype(np.uint8))
    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).ellipse((320, 230, 520, 430), fill=110)
    bg = Image.composite(Image.fromarray((a * 0.6).astype(np.uint8)), bg, shadow)
    ImageDraw.Draw(bg).ellipse((300, 200, 500, 400), fill=(240, 200, 95))
    v = photo.views(bg)
    assert v["box"] is not None
    x0, y0, x1, y1 = v["box"]
    assert x1 < 0.72 and y1 < 0.76   # the shadow (down-right) is not part of the piece


def test_busy_background_gives_no_cutout():
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).resize((800, 600))
    v = photo.views(im)
    assert v["mask"] is None and "piece" not in v and v["box"] is None
    assert v["full"].size[0] == v["full"].size[1] == 800


def test_crop_reaching_outside_the_photo_is_filled():
    im = Image.new("RGB", (100, 100), (10, 20, 30))
    out = photo.square_crop(im, (-50, -50, 50, 50), fill=(255, 255, 255))
    assert out.getpixel((10, 10)) == (255, 255, 255) and out.getpixel((80, 80)) == (10, 20, 30)


# ---- API (fake engine: no model) ---------------------------------------------

class FakePQ:
    def __init__(self, n):
        self.n = n


class FakeEngine:
    by_uid = {}

    def __init__(self):
        self.reads = 0
        self.calls = []

    def read_photo(self, im):
        self.reads += 1
        return FakePQ(self.reads)

    def search_photo(self, pq, prompt="", category=None, metal=None, page=0):
        self.calls.append((pq.n, prompt, category, metal, page))
        return {"query": {}, "dna": {}, "notes": [], "page": page, "has_more": False, "matches_in_filter": 0,
                "results": []}


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    users = {"alice": {"uid": "u-alice", "name": "A", "email": "a@x.in"},
             "bob": {"uid": "u-bob", "name": "B", "email": "b@x.in"}}
    monkeypatch.setattr(auth, "read_session", lambda cookie: users.get(cookie))

    async def approved(uid, fresh=False):
        return True
    monkeypatch.setattr(auth, "is_approved", approved)
    fake = FakeEngine()
    monkeypatch.setattr(server, "engine", fake)
    monkeypatch.setattr(server, "_photos", type(server._photos)())
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    c = TestClient(server.app)
    c.fake = fake
    return c


def as_user(c, who):
    from jewelsearch import auth
    c.cookies.clear()
    c.cookies.set(auth.COOKIE, who)
    return c


def post(c, who, body):
    return as_user(c, who).post("/api/photo-search", json=body)


def test_photo_search_reads_once_then_reuses_the_token(client):
    img = data_url(jpeg(piece_on((25, 30, 60), (225, 185, 90))))
    r = post(client, "alice", {"image": img, "q": "rose gold"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    from jewelsearch import auth
    r2 = as_user(client, "alice").get(f"/api/photo-search/{token}", params={"q": "no stones", "page": 1})
    assert r2.status_code == 200
    assert client.fake.reads == 1   # the photo was read (and embedded) only once
    assert client.fake.calls == [(1, "rose gold", None, None, 0), (1, "no stones", None, None, 1)]


def test_token_belongs_to_its_user(client):
    from jewelsearch import auth
    token = post(client, "alice", {"image": data_url(jpeg(Image.new("RGB", (100, 100), "white")))}).json()["token"]
    r = as_user(client, "bob").get(f"/api/photo-search/{token}")
    assert r.status_code == 410


def test_expired_token_asks_for_the_photo_again(client, monkeypatch):
    from jewelsearch import auth, server
    token = post(client, "alice", {"image": data_url(jpeg(Image.new("RGB", (100, 100), "white")))}).json()["token"]
    monkeypatch.setattr(server, "PHOTO_TTL", -1)
    assert as_user(client, "alice").get(f"/api/photo-search/{token}").status_code == 410


def test_photo_search_requires_sign_in(client):
    r = client.post("/api/photo-search", json={"image": "data:image/jpeg;base64,AAAA"})
    assert r.status_code == 401


def test_photo_search_requires_json(client):
    from jewelsearch import auth
    r = as_user(client, "alice").post("/api/photo-search", content=b"x", headers={"content-type": "text/plain"})
    assert r.status_code == 415


def test_bad_photo_gives_a_readable_error(client):
    r = post(client, "alice", {"image": data_url(b"not an image")})
    assert r.status_code == 400 and "photo" in r.json()["detail"].lower()


def test_bad_filter_is_refused(client):
    r = post(client, "alice", {"image": data_url(jpeg(Image.new("RGB", (100, 100), "white"))), "category": "tiara"})
    assert r.status_code == 400


# ---- the piece finder's map -> box (no model) -------------------------------------------

def test_heat_box_takes_the_piece_not_a_stray_patch():
    heat = np.zeros((24, 24))
    heat[2, 2] = 0.75                                        # a lone weak patch
    heat[10:13, 14:16] = 0.95                                # the piece
    heat[11, 16] = heat[12, 17] = 0.9                        # joined diagonally (a band)
    assert photo.heat_box(heat, 0.7) == (10, 14, 13, 18)
    assert photo.heat_box(heat, 0.99) is None


def test_heat_box_keeps_both_earrings_of_a_pair():
    heat = np.zeros((24, 24))
    heat[8:12, 4:6] = 0.95                                   # left earring
    heat[8:11, 17:19] = 0.9                                  # right earring, a little less sure
    assert photo.heat_box(heat, 0.7) == (8, 4, 12, 19)


def test_grid_to_photo_pads_to_a_square_in_photo_pixels():
    # a 600 x 400 photo squared to 600 x 600, from y = -100; cells are 25 px
    box = photo.grid_to_photo((8, 4, 10, 8), 24, (0, -100, 600, 500))
    assert box[2] - box[0] == box[3] - box[1] == round(100 * 1.12)
    assert (box[0] + box[2]) / 2 == 150 and (box[1] + box[3]) / 2 == 125


def test_square_full_reports_its_box():
    im = Image.new("RGB", (300, 200), (10, 20, 30))
    sq, box, fill = photo.square_full(im)
    assert sq.size == (300, 300) and box == (0, -50, 300, 250) and fill == (10, 20, 30)


def test_heat_on_mask_reads_the_map_under_the_cut_out():
    heat = np.zeros((24, 24))
    heat[:, 12:] = 1.0                                       # right half is jewellery
    mask = np.zeros((100, 150), bool)                        # a 300 x 200 photo at analysis size
    mask[:, 100:] = True                                     # cut-out on the right third
    assert photo.heat_on_mask(heat, (0, -50, 300, 250), (300, 200), mask) > 0.95
    mask[:] = False
    mask[:, :40] = True
    assert photo.heat_on_mask(heat, (0, -50, 300, 250), (300, 200), mask) < 0.05
