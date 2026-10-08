import asyncio

import pytest

from jewelsearch import orders
from tests.fake_rest import FakeRest

CUSTOMER = {"uid": "0f8fad5b-d9cb-469f-a165-70867728950e", "email": "asha@example.com", "name": "Asha Patel"}
JEWELER = {"id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427", "name": "Workshop Ravi", "role": "jeweler"}
RING = {"design_id": "DDLR-092", "folders": ["01/Loat - 3"], "category": "ring", "metals": ["yellow_gold", "white_gold"],
        "stone_shape": "round"}
PENDANT = {**RING, "design_id": "GPD-0005", "category": "pendant"}
DETAIL = {"uid": 41, "thumb": "/crops/a.webp", "thumbs_by_metal": {"yellow_gold": "/crops/y.webp"},
          "views": {"yellow_gold": [{"thumb": f"/crops/y{i}.webp", "full": "x"} for i in range(6)]}}
SPECS = {"purities_for": {"yellow_gold": ["14k", "18k", "22k"], "white_gold": ["14k", "18k"]},
         "gold": {"source": "card", "by_purity": {"14k": 3.4, "18k": 4.0, "22k": 4.56}}, "stones": None}
QUOTE = {"gold_weight_g": 4.0, "gold_rate": 7000, "making_mode": "per_g", "making_value": 800,
         "diamond_value": 25000, "diamond_quality": "GH VS", "other_label": "Hallmark", "other_value": 45,
         "gst_pct": 3, "note": ""}


def ring_order(**kw):
    args = dict(metal="yellow_gold", purity="18k", ring_size=12, quantity=1, phone="+91 98765 43210", note="")
    return orders.build_order(CUSTOMER, RING, DETAIL, SPECS, **{**args, **kw})


# ---------------------------------------------------------------- placing an order

@pytest.mark.parametrize("raw,clean", [("+91 98765 43210", "+919876543210"), ("+44 (20) 7946-0958", "+442079460958")])
def test_phone_numbers_keep_their_country_code(raw, clean):
    assert orders.clean_phone(raw) == clean


@pytest.mark.parametrize("raw", ["9876543210", "+91 12", "call me", "", None])
def test_phone_without_country_code_is_refused_not_guessed(raw):
    with pytest.raises(orders.OrderError):
        orders.clean_phone(raw)


def test_order_freezes_the_design_and_the_choices():
    row = ring_order(quantity=2, note="  engrave A\n\n\n\nplease ")
    assert row["design_key"] == "DDLR-092|01/Loat - 3" and row["status"] == "request"
    assert row["ring_size_in"] == 12 and row["quantity"] == 2
    assert row["customer_phone"] == "+919876543210" and row["customer_note"] == "engrave A\n\nplease"
    snap = row["snapshot"]
    assert snap["gold_weight_g"] == 4.0 and snap["gold_source"] == "card"
    assert snap["selection"]["ring_size"]["in"] == 12 and snap["selection"]["purity_label"] == "18K"
    assert snap["design"]["views"] == [f"/crops/y{i}.webp" for i in range(4)]
    assert row["thumb"] == "/crops/y.webp"


@pytest.mark.parametrize("kw,msg", [
    ({"ring_size": None}, "ring size"), ({"ring_size": 40}, "ring size"),
    ({"metal": "rose_gold"}, "metal colour"), ({"metal": "white_gold", "purity": "22k"}, "purity"),
    ({"quantity": 0}, "between"), ({"phone": "98765 43210"}, "country code"),
])
def test_orders_offered_only_as_the_buy_page_allows(kw, msg):
    with pytest.raises(orders.OrderError) as e:
        ring_order(**kw)
    assert msg in e.value.message


def test_ring_size_is_ignored_for_other_pieces():
    row = orders.build_order(CUSTOMER, PENDANT, DETAIL, SPECS, metal="yellow_gold", purity="22k", ring_size=12,
                             quantity=1, phone="+919876543210", note="")
    assert row["ring_size_in"] is None and row["snapshot"]["selection"]["ring_size"] is None


def test_no_weight_on_file_stays_empty():
    row = orders.build_order(CUSTOMER, RING, DETAIL, {**SPECS, "gold": None}, metal="yellow_gold", purity="18k",
                             ring_size=12, quantity=1, phone="+919876543210", note="")
    assert row["snapshot"]["gold_weight_g"] is None and row["snapshot"]["gold_source"] is None


# ---------------------------------------------------------------- money

def test_bill_adds_up_the_jewelers_figures():
    b = orders.bill(orders.build_quote(QUOTE, "Ravi"))
    assert (b["gold"], b["making"], b["diamonds"], b["other"]) == (28000, 3200, 25000, 45)
    assert b["subtotal"] == 56245 and b["gst"] == 1687.35 and b["total"] == 57932


@pytest.mark.parametrize("mode,value,making", [("per_g", 500, 2000), ("percent", 12, 3360), ("fixed", 1500, 1500)])
def test_making_charge_modes(mode, value, making):
    q = orders.build_quote({**QUOTE, "making_mode": mode, "making_value": value}, "Ravi")
    assert orders.bill(q)["making"] == making


def test_final_bill_follows_the_weighed_gold():
    q = orders.build_quote(QUOTE, "Ravi")
    b = orders.bill(q, weight_g=4.21)
    assert b["gold"] == 29470 and b["making"] == 3368 and b["diamonds"] == 25000   # stones stay as quoted


def test_total_rounds_half_up_to_the_rupee():
    q = orders.build_quote({**QUOTE, "gold_weight_g": 1, "gold_rate": 100.5, "making_mode": "fixed", "making_value": 0,
                            "diamond_value": 0, "other_value": 0, "gst_pct": 0}, "Ravi")
    assert orders.bill(q)["total"] == 101


@pytest.mark.parametrize("bad", [{"gold_rate": 0}, {"gold_weight_g": -1}, {"gst_pct": 40}, {"making_mode": "?"},
                                 {"making_mode": "percent", "making_value": 150}, {"gold_rate": float("nan")},
                                 {"diamond_value": True}])
def test_quote_values_are_checked(bad):
    with pytest.raises(orders.OrderError):
        orders.build_quote({**QUOTE, **bad}, "Ravi")


# ---------------------------------------------------------------- stages

def order_in(status, **kw):
    return {"id": 7, "status": status, "quote": None, "payments": [], "actual_weight_g": None, **kw}


def test_orders_move_one_stage_at_a_time():
    with pytest.raises(orders.OrderError):
        orders.check_move(order_in("request"), "approved", "")
    orders.check_move(order_in("approved"), "quoted", "")   # one back is fine


@pytest.mark.parametrize("status,to,missing", [
    ("request", "quoted", "price"), ("approved", "advance_paid", "advance"), ("quality_check", "final_bill", "weight"),
])
def test_a_stage_needs_its_figures_first(status, to, missing):
    with pytest.raises(orders.OrderError) as e:
        orders.check_move(order_in(status), to, "")
    assert missing in e.value.message
    assert missing in orders.view(order_in(status))["next"]["blocked"]


def test_cancelling_needs_a_reason_and_is_final():
    with pytest.raises(orders.OrderError):
        orders.check_move(order_in("in_production"), "cancelled", "")
    orders.check_move(order_in("in_production"), "cancelled", "customer changed mind")
    with pytest.raises(orders.OrderError):
        orders.check_move(order_in("delivered"), "cancelled", "late")
    with pytest.raises(orders.OrderError):
        orders.check_move(order_in("cancelled"), "request", "")


def test_search_box_cannot_inject_filter_syntax():
    f = orders.search_filter('asha),status.eq.(delivered"')
    assert f.count("(") == 1 and f.count(")") == 1 and '"' not in f.replace('."*', "").replace('*"', "")
    assert "id.eq.12" in orders.search_filter("DF-00012") and "id.eq.12" in orders.search_filter("df12")
    assert orders.search_filter("  ") is None


# ---------------------------------------------------------------- the whole flow, against a fake Supabase

@pytest.fixture
def db(monkeypatch):
    fake = FakeRest()
    monkeypatch.setattr(orders, "_call", fake)
    return fake


def run(coro):
    return asyncio.run(coro)


def test_order_from_request_to_delivery(db):
    customer = {"id": CUSTOMER["uid"], "name": CUSTOMER["name"], "role": "customer"}
    o = run(orders.create(ring_order(), customer))
    assert orders.order_no(o["id"]) == "DF-00001"

    listed = run(orders.list_orders("active", "", 50))
    assert listed["counts"]["active"] == 1 and listed["items"][0]["order_no"] == "DF-00001"
    assert run(orders.list_orders("active", "asha", 50))["items"] and not run(orders.list_orders("delivered", "", 50))["items"]

    o = run(orders.save_quote(o["id"], QUOTE, o["updated_at"], JEWELER))
    assert o["quoted_total"] == 57932
    for stage in ("quoted", "approved"):
        o = run(orders.move(o["id"], stage, "", o["updated_at"], JEWELER))
    with pytest.raises(orders.OrderError, match="locked"):
        run(orders.save_quote(o["id"], QUOTE, o["updated_at"], JEWELER))

    o = run(orders.add_payment(o["id"], 20000, "upi", "UTR 1234", o["updated_at"], JEWELER))
    for stage in ("advance_paid", "in_production", "quality_check"):
        o = run(orders.move(o["id"], stage, "", o["updated_at"], JEWELER))
    o = run(orders.set_weight(o["id"], 4.21, "after polish", o["updated_at"], JEWELER))
    o = run(orders.move(o["id"], "final_bill", "", o["updated_at"], JEWELER))

    t = orders.view(o)["totals"]
    assert t["quote"]["total"] == 57932 and t["final"]["gold"] == 29470 and t["paid"] == 20000
    assert t["due"] == t["final"]["total"] - 20000
    for stage in ("ready", "delivered"):
        o = run(orders.move(o["id"], stage, "", o["updated_at"], JEWELER))
    assert orders.view(o)["can"] == {"quote": False, "weight": False, "pay": False, "cancel": False}

    run(orders.add_note(o["id"], "Collected by her brother", JEWELER))
    kinds = [e["kind"] for e in run(orders.events(o["id"]))]
    assert kinds[0] == "note" and kinds[-1] == "created" and kinds.count("stage") == 8 and "weight" in kinds


def test_a_stale_page_cannot_overwrite_a_newer_change(db):
    o = run(orders.create(ring_order(), {"id": CUSTOMER["uid"], "name": "Asha", "role": "customer"}))
    seen = o["updated_at"]
    run(orders.save_quote(o["id"], QUOTE, seen, JEWELER))                 # one jeweler saves a price
    with pytest.raises(orders.OrderError) as e:                             # another still has the old page open
        run(orders.move(o["id"], "cancelled", "duplicate", seen, JEWELER))
    assert e.value.status == 409
    assert run(orders.get(o["id"]))["status"] == "request"


def test_payments_can_be_removed_and_are_logged(db):
    o = run(orders.create(ring_order(), {"id": CUSTOMER["uid"], "name": "Asha", "role": "customer"}))
    o = run(orders.add_payment(o["id"], 5000, "cash", "", o["updated_at"], JEWELER))
    pid = o["payments"][0]["id"]
    o = run(orders.remove_payment(o["id"], pid, o["updated_at"], JEWELER))
    assert o["payments"] == [] and orders.paid(o) == 0
    kinds = [e["kind"] for e in run(orders.events(o["id"]))]
    assert kinds[:2] == ["payment_removed", "payment"]


def test_missing_tables_say_how_to_set_them_up(monkeypatch):
    monkeypatch.setattr(orders, "_call", FakeRest(tables=()))
    with pytest.raises(orders.OrderError) as e:
        run(orders.list_orders("active", "", 10))
    assert e.value.status == 503 and "orders.sql" in e.value.message
