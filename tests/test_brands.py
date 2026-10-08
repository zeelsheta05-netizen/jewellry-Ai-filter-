"""Brand designs (jewelsearch/brands.py): reading another jeweller's product page,
our version and its pricing, orders through "Buy with us", and the whole team
flow through the server. No internet: the web is a stand-in."""
import io
import json
import time

import pytest
from PIL import Image

from jewelsearch import brands, linksearch, orders, storage, webarchive
from tests.fake_rest import FakeRest


@pytest.fixture(autouse=True)
def dataset(tmp_path, monkeypatch):
    """Fetched pages are staged in a test folder and stored in a test dataset, never the real ones."""
    monkeypatch.setattr(webarchive, "STAGING", tmp_path / "staging")
    root = tmp_path / "dataset"
    root.mkdir()
    before = storage._current
    storage.use(storage.LocalStorage(root))
    yield root
    storage.use(before)

PAGE = """<!doctype html><html><head><title>Aurora Halo Ring | Shop</title>
<meta property="og:site_name" content="Sparkle Co">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"Product","name":"Aurora Halo Ring",
 "image":["https://shop.example/img/ring-1.png"],"sku":"AH-101","brand":{"@type":"Brand","name":"Sparkle Co"},
 "description":"Set in 18 KT Yellow Gold(2.150 g) with diamonds (0.280 ct ,FG-SI)",
 "offers":{"@type":"Offer","price":"45,990","priceCurrency":"INR"}}</script></head>
<body>
<nav><a>Gold</a><span>Gold Rings</span><a>Gemstone</a><span>SHOP BY PRICE</span></nav>
<header><div>Diamond</div><div>Shop by occasion</div></header>
<div class="specs">
  <div><span>Metal Purity</span><span>18 KT</span></div>
  <div><span>Diamond Clarity</span><span>SI</span></div>
  <div><span>No. of Diamonds</span><span>24</span></div>
  <div><span>Ring Width</span>:<span>2.1 mm</span></div>
  <p>Occasion: Everyday</p>
</div>
<table><tr><th>Component</th><th>Weight</th><th>Rate</th><th>Value</th></tr>
  <tr><td>18 KT Yellow Gold</td><td>2.150 g</td><td>₹ 7000 /g</td><td>₹ 15050</td></tr>
  <tr><td>Making Charges</td><td>2.150 g</td><td>₹ 900 /g</td><td>₹ 1935</td></tr>
  <tr><td>GH VS round - 24 No.s</td><td>0.280 ct</td><td>-</td><td>₹ 26000</td></tr></table>
<img src="/img/ring-2.png" alt="Aurora Halo Ring side" width="800" height="800">
<img src="/img/logo.png" alt="logo">
<footer><div>Gold</div><div>Our stores</div></footer>
</body></html>"""


def png(side=600, color=(250, 240, 220)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (side, side), color).save(buf, "PNG")
    return buf.getvalue()


class Web:
    """linksearch.fetch, for a few known addresses."""

    def __init__(self, pages: dict):
        self.pages, self.seen, self.accepts = pages, [], {}

    def __call__(self, url, accept, max_bytes, deadline):
        self.seen.append(url)
        self.accepts[url] = accept
        if url not in self.pages:
            raise linksearch.LinkError("The website refused to open the page (error 404).")
        ctype, data = self.pages[url]
        return linksearch.Fetched(url, ctype, data)


SHOP = {
    "https://shop.example/p/aurora": ("text/html", PAGE.encode()),
    "https://shop.example/img/ring-1.png": ("image/png", png(600, (250, 240, 220))),
    "https://shop.example/img/ring-2.png": ("image/png", png(800, (240, 230, 250))),
    "https://shop.example/img/logo.png": ("image/png", png(60)),
}


# ---------------------------------------------------------------- reading a page

def test_a_product_page_gives_its_details_without_menu_noise():
    s = brands.read_product("https://shop.example/p/aurora", PAGE, time.monotonic() + 5)
    o = s.original
    assert (o["title"], o["brand"], o["sku"], o["price"], o["currency"]) == ("Aurora Halo Ring", "Sparkle Co", "AH-101", 45990, "INR")
    specs = {x["label"]: x["value"] for x in o["specs"]}
    assert specs["Metal Purity"] == "18 KT" and specs["No. of Diamonds"] == "24" and specs["Ring Width"] == "2.1 mm"
    assert specs["Occasion"] == "Everyday"
    values = " ".join(specs.values()).lower()
    assert "shop by" not in values and "gold rings" not in values and "stores" not in values   # menus and footers
    assert "Making Charges" not in specs     # a wide table's cells aren't paired up...
    assert o["tables"][0]["rows"][1][:2] == ["18 KT Yellow Gold", "2.150 g"]   # ...the table is kept as a table
    assert s.picture_urls[:2] == ["https://shop.example/img/ring-1.png", "https://shop.example/img/ring-2.png"]


def test_key_figures_come_from_labels_tables_and_stated_figures():
    k = brands.read_product("https://shop.example/p/aurora", PAGE, time.monotonic() + 5).original["key"]
    assert k["purity"] == "18K" and k["metal"] == "yellow_gold" and k["gold_g"] == 2.15
    assert k["diamond_ct"] == 0.28 and k["diamond_count"] == 24 and k["diamond_quality"] == "FG-SI"


def test_plated_and_silver_pieces_are_not_read_as_gold():
    page = """<html><body><div><span>Plating</span><span>18k Gold Plated</span></div>
      <div><span>Weight</span><span>1.7 gm</span></div></body></html>"""
    k = brands.read_product("https://x.example/p", page, time.monotonic() + 5).original["key"]
    assert k["metal"] == "gold plated" and k["purity"] is None and k["gold_g"] is None
    silver = brands.key_figures({"title": "Star ring", "specs": [{"label": "Metal", "value": "925 Silver"},
                                                                {"label": "Net Weight", "value": "2.0 g"}]})
    assert silver["metal"] == "silver" and silver["gold_g"] is None


def test_shopify_product_json_adds_price_mrp_pictures_and_tags(monkeypatch):
    page = """<html><head><script src="https://cdn.shopify.com/x.js"></script></head><body><h1>Star</h1></body></html>"""
    product = {"title": "Star Ring", "vendor": "Giva", "price": 109900, "compare_at_price": 329900,
               "tags": ["Metal_925 Silver", "Stone_Zircon", "Price_Change_1", "diwali_2024"],
               "options": [{"name": "Size", "values": ["10", "12"]}],
               "images": ["//cdn.shopify.com/files/R1_1.jpg?v=1"], "variants": [{"sku": "R0126"}],
               "description": "<p>Ring Diameter: 1.66 cm</p>"}
    web = Web({"https://giva.example/products/star-ring.js": ("text/javascript", json.dumps(product).encode())})
    monkeypatch.setattr(linksearch, "fetch", web)
    o = brands.read_product("https://giva.example/products/star-ring?variant=4", page, time.monotonic() + 5)
    orig = o.original
    assert (orig["price"], orig["mrp"], orig["sku"], orig["brand"], orig["via"]) == (1099, 3299, "R0126", "Giva", "shopify")
    specs = {x["label"]: x["value"] for x in orig["specs"]}
    assert specs == {"Metal": "925 Silver", "Stone": "Zircon", "Size": "10, 12", "Ring Diameter": "1.66 cm"}
    assert o.picture_urls[0] == "https://cdn.shopify.com/files/R1_1.jpg?v=1"


def test_fetching_keeps_product_pictures_and_drops_icons(monkeypatch):
    monkeypatch.setattr(linksearch, "fetch", Web(SHOP))
    res = brands.fetch_link("see https://shop.example/p/aurora", "team-1")
    assert [p["width"] for p in res["pictures"]] == [600, 800]   # the 60 px logo is left out
    assert res["pictures"][0]["preview"].startswith("data:image/jpeg;base64,")
    pf = res["prefill"]
    assert pf["category"] == "ring" and pf["metals"] == ["yellow_gold"] and pf["purities"] == ["14k", "18k"]
    assert pf["gold_g"] == {"14k": 1.827, "18k": 2.15}            # job-card ratios from the stated 18K weight
    assert pf["stones"][0]["count"] == 24 and pf["stones"][0]["carat"] == 0.28
    d = brands.take_draft(res["token"], "team-1")
    assert webarchive.read(d.folder, d.pictures[0]["file"]) == SHOP["https://shop.example/img/ring-1.png"][1]   # as downloaded
    with pytest.raises(brands.BrandError):
        brands.take_draft(res["token"], "someone-else")


def test_a_picture_link_or_a_blocked_shop_says_what_to_do(monkeypatch):
    monkeypatch.setattr(linksearch, "fetch", Web(SHOP))
    with pytest.raises(brands.BrandError) as e:
        brands.fetch_link("https://shop.example/img/ring-1.png", "t")
    assert "product page" in e.value.message
    with pytest.raises(brands.BrandError) as e:
        brands.fetch_link("https://shop.example/p/missing", "t")
    assert "refused" in e.value.message


# ---------------------------------------------------------------- pictures at their best, and the archive

@pytest.mark.parametrize("small,full", [
    ("https://www.giva.co/cdn/shop/files/R0126_1.jpg?v=17&width=1920", "https://www.giva.co/cdn/shop/files/R0126_1.jpg?v=17"),
    ("https://photos.melorra.com/fit-in/515x515/dev/catalogue/a_P_580.png", "https://photos.melorra.com/dev/catalogue/a_P_580.png"),
    ("https://res.cloudinary.com/x/image/upload/w_500,h_500,c_fill/v1/ring.jpg", "https://res.cloudinary.com/x/image/upload/v1/ring.jpg"),
    ("https://shop.example/media/catalog/product/cache/1/image/0f3a9c2b8d7e6f5a4b3c2d1e0f9a8b7c/r/i/ring.jpg",
     "https://shop.example/media/catalog/product/r/i/ring.jpg"),
    ("https://blog.example/wp-content/uploads/ring-300x300.jpg", "https://blog.example/wp-content/uploads/ring.jpg"),
    ("https://cdn.shopify.com/s/files/1/ring_600x.jpg?v=3", "https://cdn.shopify.com/s/files/1/ring.jpg?v=3"),
    ("https://www.tanishq.co.in/images/ring.jpg?sw=480&sh=480&sm=fit", "https://www.tanishq.co.in/images/ring.jpg"),
])
def test_resized_picture_links_lead_to_the_full_size_file(small, full):
    assert brands.upgrade_urls(small) == [full, small]


def test_a_plain_picture_link_is_tried_as_it_is():
    url = "https://cdn.caratlane.com/media/catalog/product/B/R/BR00895-SSS300_1_lar.jpg"
    assert brands.upgrade_urls(url) == [url]


def test_the_full_size_file_is_preferred_and_webp_copies_are_not_asked_for(monkeypatch):
    big, small = png(1800), png(515)
    web = Web({"https://img.example/full/ring.png": ("image/png", big),
               "https://img.example/fit-in/515x515/full/ring.png": ("image/png", small)})
    monkeypatch.setattr(linksearch, "fetch", web)
    got = brands.download_pictures(["https://img.example/fit-in/515x515/full/ring.png"], time.monotonic() + 5)
    assert [(p.width, p.data) for p in got] == [(1800, big)]
    assert "webp" not in web.accepts["https://img.example/full/ring.png"]
    # no full-size copy on the server: the page's own link
    web.pages.pop("https://img.example/full/ring.png")
    got = brands.download_pictures(["https://img.example/fit-in/515x515/full/ring.png"], time.monotonic() + 5)
    assert [p.width for p in got] == [515]


def test_the_same_photo_at_two_sizes_is_kept_once_at_the_larger_size(monkeypatch):
    def photo_png(side):
        im = Image.new("RGB", (side, side), "white")
        for x in range(side // 4, side // 2):
            for y in range(side // 3, side // 2):
                im.putpixel((x, y), (200, 150, 40))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()
    pages = {"https://a.example/s.png": ("image/png", photo_png(400)), "https://a.example/l.png": ("image/png", photo_png(900)),
             "https://a.example/other.png": ("image/png", png(500, (20, 30, 200)))}
    monkeypatch.setattr(linksearch, "fetch", Web(pages))
    got = brands.download_pictures(list(pages), time.monotonic() + 5)
    assert sorted(p.width for p in got) == [500, 900]


def test_one_ring_in_three_metal_colours_stays_three_photos(monkeypatch):
    def ring(color):
        im = Image.new("RGB", (600, 600), "white")
        for x in range(200, 400):
            for y in range(250, 350):
                im.putpixel((x, y), color)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()
    pages = {f"https://a.example/{n}.png": ("image/png", ring(c))
             for n, c in (("YL", (212, 175, 55)), ("WH", (200, 200, 205)), ("RS", (205, 140, 120)))}
    monkeypatch.setattr(linksearch, "fetch", Web(pages))
    assert len(brands.download_pictures(list(pages), time.monotonic() + 5)) == 3


def test_every_fetch_is_stored_in_the_dataset_with_its_page_and_details(monkeypatch, dataset):
    monkeypatch.setattr(linksearch, "fetch", Web(SHOP))
    res = brands.fetch_link("https://shop.example/p/aurora", "team-1", "Ravi")
    folder = res["folder"]
    assert folder.startswith("Web Designs/shop.example/") and "_aurora-halo-ring_" in folder
    staged = webarchive.STAGING / folder
    assert sorted(f.name for f in staged.iterdir()) == ["01.png", "02.png", "design.json", "page.html", "preview.jpg"]
    assert webarchive.push(folder) is True
    stored = dataset / folder
    assert (stored / "01.png").read_bytes() == SHOP["https://shop.example/img/ring-1.png"][1]
    assert (stored / "page.html").read_bytes() == PAGE.encode()
    info = json.loads((stored / "design.json").read_text())
    assert info["original"]["price"] == 45990 and info["scraped_by"] == "Ravi"
    assert info["pictures"][1]["source"] == "https://shop.example/img/ring-2.png" and len(info["pictures"][1]["sha256"]) == 64
    assert not staged.exists()                                        # the local copy is gone once stored
    assert webarchive.read(folder, "02.png") == SHOP["https://shop.example/img/ring-2.png"][1]   # read back from the dataset


def test_without_the_dataset_storage_the_files_wait_and_go_later(monkeypatch, tmp_path, dataset):
    monkeypatch.setattr(linksearch, "fetch", Web(SHOP))
    folder = brands.fetch_link("https://shop.example/p/aurora", "t")["folder"]
    storage.use(storage.LocalStorage(tmp_path / "unplugged"))          # not there: not ready
    assert webarchive.push(folder) is False and (webarchive.STAGING / folder / "01.png").is_file()
    assert webarchive.staged() == [folder]
    storage.use(storage.LocalStorage(dataset))
    assert webarchive.push(folder) is True and (dataset / folder / "01.png").is_file() and webarchive.staged() == []


# ---------------------------------------------------------------- our version

OURS = {"name": "Aurora Halo, our version", "design_no": "DF-B-1", "category": "ring",
        "metals": ["yellow_gold", "rose_gold"], "purities": ["14k", "18k"], "gold_g": {"14k": 2.0, "18k": 2.35},
        "stones": [{"shape": "Round", "size": "1.3 mm", "count": 22, "carat": 0.25, "setting": "Pavé"}],
        "diamond_quality": "EF VVS", "dimensions": [{"label": "Width", "value": "2.2 mm"}], "ring_size_in": 12,
        "pricing": {"gold_rate_24k": 10000, "making_mode": "per_g", "making_value": 800, "diamond_value": 20000,
                    "other_label": "Hallmark", "other_value": 45, "gst_pct": 3}, "note": "Slimmer band."}


def test_our_price_follows_the_gold_content_of_each_purity():
    assert brands.purity_rate(10000, "18k") == 7500 and brands.purity_rate(10000, "14k") == 5850
    ours = brands.clean_ours(OURS)
    p18 = ours["prices"]["18k"]
    # 2.35 g x 7500 = 17625; making 2.35 x 800 = 1880; + 20000 + 45 = 39550; GST 3% = 1186.5
    assert (p18["gold"], p18["making"], p18["subtotal"], p18["gst"], p18["total"]) == (17625, 1880, 39550, 1186.5, 40737)
    assert ours["prices"]["14k"]["gold_rate"] == 5850


@pytest.mark.parametrize("change,msg", [
    ({"category": "watch"}, "category"), ({"metals": []}, "metal colour"),
    ({"purities": ["22k"], "gold_g": {"22k": 3}}, "22K"), ({"gold_g": {"18k": 2.35}}, "14K"),
    ({"stones": [{"shape": "Round", "count": 0, "carat": 0.1}]}, "number of Round"),
    ({"stones": [{"shape": "Round", "count": 3}]}, "carats"),
    ({"ring_size_in": 40}, "ring size"), ({"pricing": {**OURS["pricing"], "gold_rate_24k": None}}, "24K"),
    ({"pricing": {**OURS["pricing"], "making_mode": "percent", "making_value": 140}}, "percent"),
    ({"name": " "}, "name"),
])
def test_our_details_are_checked(change, msg):
    with pytest.raises(brands.BrandError) as e:
        brands.clean_ours({**OURS, **change})
    assert msg in e.value.message


def test_empty_stone_rows_mean_a_plain_design_and_size_is_only_for_rings():
    ours = brands.clean_ours({**OURS, "category": "pendant", "stones": [{"shape": "", "count": None}]})
    assert ours["stones"] == [] and ours["ring_size_in"] is None
    specs = brands.specs_for_buy(ours)
    assert specs["stones"] == {"source": "ours", "counted": True, "groups": [], "count": 0, "carat": 0}


def _row(ours=None):
    return {"id": 7, "status": "listed", "site": "shop.example", "source_url": "https://shop.example/p/aurora",
            "original": {"title": "Aurora Halo Ring", "brand": "Sparkle Co", "price": 45990.0, "currency": "INR"},
            "ours": brands.clean_ours(ours or OURS), "pictures": [{"file": "a" * 32 + ".png", "width": 600, "height": 600}]}


def test_buy_with_us_gives_the_buy_page_its_usual_shape():
    b = brands.buy_view(_row())
    assert b["uid"] is None and b["brand_id"] == 7 and b["design_id"] == "DF-B-1"
    assert b["specs"]["purities_for"] == {"yellow_gold": ["14k", "18k"], "rose_gold": ["14k", "18k"]}
    assert b["specs"]["gold"]["by_purity"] == {"14k": 2.0, "18k": 2.35} and b["specs"]["ring_size"]["in"] == 12
    assert b["prices"]["18k"] == 40737 and b["views"]["rose_gold"][0]["full"] == "/brand-media/" + "a" * 32 + ".png"


def test_a_brand_order_is_a_normal_order_with_the_listing_frozen_in():
    user = {"uid": "0f8fad5b-d9cb-469f-a165-70867728950e", "email": "asha@example.com", "name": "Asha"}
    row = brands.build_order(user, _row(), metal="rose_gold", purity="18k", ring_size=13, quantity=2,
                             phone="+91 98765 43210", note="")
    assert row["kind"] == "brand" and row["design_uid"] == 7 and row["design_key"] == "brand:7"
    snap = row["snapshot"]
    assert snap["gold_weight_g"] == 2.35 and snap["gold_source"] == "ours" and snap["selection"]["ring_size"]["in"] == 13
    assert snap["brand"]["listed_price"] == 40737 and snap["brand"]["seller_price"] == 45990
    assert snap["brand"]["quote"]["gold_rate"] == 7500 and snap["brand"]["seller_url"] == "https://shop.example/p/aurora"
    with pytest.raises(orders.OrderError):   # checked like any order: 22K isn't offered here
        brands.build_order(user, _row(), metal="yellow_gold", purity="22k", ring_size=13, quantity=1,
                           phone="+919876543210", note="")


def test_the_seller_link_must_be_a_web_link():
    row = _row()
    row["source_url"] = "javascript:alert(1)"
    assert brands.card(row)["seller_url"] is None


# ---------------------------------------------------------------- the whole flow, through the server

@pytest.fixture
def app(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    db = FakeRest(tables=("brand_designs", "orders", "order_events", "web_designs"))
    monkeypatch.setattr(webarchive, "_call", db)
    monkeypatch.setattr(brands, "_call", db)
    monkeypatch.setattr(orders, "_call", db)
    monkeypatch.setattr(brands, "PICTURES", tmp_path / "pictures")
    monkeypatch.setattr(linksearch, "fetch", Web(SHOP))
    who = {"user": {"uid": "0f8fad5b-d9cb-469f-a165-70867728950e", "email": "ravi@shop.in", "name": "Ravi"}, "role": "jeweler"}
    monkeypatch.setattr(auth, "read_session", lambda c: who["user"])

    async def approved(uid, fresh=False):
        return True

    async def role(uid):
        return who["role"]
    monkeypatch.setattr(auth, "is_approved", approved)
    monkeypatch.setattr(auth, "staff_role", role)
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    # no search index in tests: order pages only pass picture links through it
    monkeypatch.setattr(server, "engine", type("E", (), {"original_url": staticmethod(lambda u: u)})())
    return TestClient(server.app), db, who


def test_team_lists_a_design_and_a_customer_buys_it_with_us(app, dataset):
    client, db, who = app
    got = client.post("/api/brand-designs/fetch", json={"url": "https://shop.example/p/aurora"})
    assert got.status_code == 200, got.text
    f = got.json()
    web = db.tables["web_designs"][0]                                 # kept in the collection...
    assert web["folder"] == f["archive"]["folder"] and web["archive_status"] == "stored"   # ...and sent (background task)
    assert (dataset / web["folder"] / "02.png").is_file() and not (webarchive.STAGING / web["folder"]).exists()
    made = client.post("/api/brand-designs", json={"token": f["token"], "pictures": [1, 0], "ours": OURS})
    assert made.status_code == 200, made.text
    new = made.json()
    assert new["redirect"] == f"/?browse=ring&new={new['id']}"     # straight to the catalogue, design listed
    row = db.tables["brand_designs"][0]
    assert row["original"]["title"] == "Aurora Halo Ring" and row["ours"]["prices"]["18k"]["total"] == 40737
    assert row["original"]["archive"] == {"folder": web["folder"], "web_design_id": web["id"]}
    assert web["brand_design_id"] == row["id"]
    items = client.get("/api/web-designs").json()["items"]
    assert items[0]["brand_design_id"] == row["id"] and items[0]["pictures"] == 2 and items[0]["largest"] == "800×800"
    assert client.get(items[0]["preview"]).headers["content-type"] == "image/jpeg"   # from the dataset storage
    assert client.post("/api/brand-designs", json={"token": f["token"], "pictures": [0], "ours": OURS}).status_code == 410

    # the catalogue and the design's page (a customer now)
    who["role"] = None
    items = client.get("/api/brand-designs?category=ring").json()["items"]
    assert [i["id"] for i in items] == [new["id"]] and items[0]["our_price_from"] == min(
        v["total"] for v in row["ours"]["prices"].values())
    assert client.get("/api/brand-designs?category=pendant").json()["items"] == []
    view = client.get(f"/api/brand-designs/{new['id']}").json()
    assert view["can_manage"] is False and view["seller_url"] == "https://shop.example/p/aurora"
    pic = client.get(view["pictures"][0]["url"])   # the first kept picture is the 800 px one (order [1, 0])
    assert pic.status_code == 200 and pic.content == SHOP["https://shop.example/img/ring-2.png"][1]
    assert pic.headers["content-type"] == "image/png"
    assert client.get("/brand-media/..%2F..%2Fsecret.png").status_code == 404

    # Buy with us: the usual order, seen by the jeweler with our listed pricing
    assert client.get(f"/api/brand-designs/{new['id']}/buy").json()["prices"]["18k"] == 40737
    placed = client.post(f"/api/brand-designs/{new['id']}/order",
                         json={"metal": "yellow_gold", "purity": "18k", "ring_size": 12, "quantity": 1, "phone": "+91 98765 43210"})
    assert placed.status_code == 200, placed.text
    who["role"] = "jeweler"
    page = client.get(f"/api/jeweler/orders/{placed.json()['id']}").json()
    assert page["brand_id"] == new["id"] and page["catalogue_uid"] is None
    assert page["order"]["kind"] == "brand" and page["order"]["snapshot"]["brand"]["listed_price"] == 40737

    # the team removes it: off the catalogue, the order keeps its copy
    assert client.delete(f"/api/brand-designs/{new['id']}").status_code == 200
    who["role"] = None
    assert client.get("/api/brand-designs").json()["items"] == []
    assert client.get(f"/api/brand-designs/{new['id']}").status_code == 404


def test_a_stored_design_can_be_listed_later_without_visiting_the_shop(app, monkeypatch):
    client, db, who = app
    f = client.post("/api/brand-designs/fetch", json={"url": "https://shop.example/p/aurora"}).json()
    monkeypatch.setattr(linksearch, "fetch", Web({}))                 # the shop is gone now
    again = client.post(f"/api/web-designs/{f['web_id']}/open")
    assert again.status_code == 200, again.text
    g = again.json()
    assert g["original"]["title"] == "Aurora Halo Ring" and len(g["pictures"]) == 2 and g["token"] != f["token"]
    made = client.post("/api/brand-designs", json={"token": g["token"], "pictures": [0], "ours": OURS})
    assert made.status_code == 200, made.text
    assert db.tables["web_designs"][0]["brand_design_id"] == made.json()["id"]


def test_without_the_collection_table_fetched_pages_are_still_stored(app, monkeypatch, dataset):
    client, db, who = app
    monkeypatch.setattr(webarchive, "_call", FakeRest(tables=("orders",)))
    f = client.post("/api/brand-designs/fetch", json={"url": "https://shop.example/p/aurora"}).json()
    assert f["archive"]["recorded"] is False and (dataset / f["archive"]["folder"] / "design.json").is_file()
    r = client.get("/api/web-designs")
    assert r.status_code == 503 and "web_designs.sql" in r.json()["detail"]


def test_only_the_team_can_fetch_list_or_remove(app):
    client, db, who = app
    who["role"] = None
    assert client.post("/api/brand-designs/fetch", json={"url": "https://shop.example/p/aurora"}).status_code == 403
    assert client.post("/api/brand-designs", json={"token": "x", "pictures": [0], "ours": OURS}).status_code == 403
    assert client.delete("/api/brand-designs/1").status_code == 403
    assert client.get("/api/web-designs").status_code == 403
    assert client.post("/api/web-designs/1/open").status_code == 403
    assert client.get("/brand-import", follow_redirects=False).status_code == 303


def test_missing_table_says_how_to_set_it_up(app, monkeypatch):
    client, db, who = app
    monkeypatch.setattr(brands, "_call", FakeRest(tables=("orders",)))
    r = client.get("/api/brand-designs")
    assert r.status_code == 503 and "brand_designs.sql" in r.json()["detail"]
