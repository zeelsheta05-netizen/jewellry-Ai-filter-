"""Orders: a customer orders a catalogue design as it is from the buy page;
jewelers (the manufacturing side) take it from request to delivery at /jeweler.

Stored in Supabase (public.orders and public.order_events, supabase/orders.sql).
Only this server reads and writes them. A customer only ever creates an order
for themselves (the user id comes from the session cookie). Jewelers and
admins see every order.

What an order holds is frozen when it is placed: the design id and folder
(index uids change when the index is rebuilt), the customer's choices, and the
figures the buy page showed (job card or CAD estimates). Rebuilding the index
or specs.json never changes an order.

Money is only ever typed in by a jeweler. The server adds up what they
entered (`bill`) but never fills in a rate or a price by itself, and the final
bill uses the weight actually measured at quality check.

Every change is checked against the order as the jeweler last saw it
(`updated_at`), so two people editing one order can't overwrite each other.
"""
from __future__ import annotations

import logging
import re
import secrets
from collections import Counter
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from . import purchase
from .auth import _call

log = logging.getLogger(__name__)
ORDERS, EVENTS = "/rest/v1/orders", "/rest/v1/order_events"

# The order's path through the workshop, one stage at a time.
STAGES = [
    ("request", "New request"),
    ("quoted", "Price sent"),
    ("approved", "Customer approved"),
    ("advance_paid", "Advance received"),
    ("in_production", "In production"),
    ("quality_check", "Quality check"),
    ("final_bill", "Final bill"),
    ("ready", "Ready / dispatched"),
    ("delivered", "Delivered"),
]
FLOW = [k for k, _ in STAGES]
LABELS = dict(STAGES) | {"cancelled": "Cancelled"}
CLOSED = {"delivered", "cancelled"}
QUOTE_EDITABLE = {"request", "quoted"}      # the price is locked once the customer approves it
WEIGHT_EDITABLE = {"quality_check"}

PAY_METHODS = {"cash": "Cash", "upi": "UPI", "bank": "Bank transfer", "card": "Card", "cheque": "Cheque", "other": "Other"}
MAKING_MODES = {"per_g": "per gram", "percent": "% of gold value", "fixed": "fixed amount"}
RING_SIZES = range(6, 27)    # Indian sizes offered on the buy page
MAX_QTY = 20

LIST_FIELDS = ("id,status,kind,design_id,category,metal,purity,ring_size_in,quantity,thumb,"
               "customer_name,customer_phone,quoted_total,created_at,updated_at")


class OrderError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def order_no(order_id: int) -> str:
    return f"DF-{int(order_id):05d}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- input checks

def clean_text(s: str | None, limit: int) -> str:
    """Printable text only; at most one blank line in a row."""
    s = re.sub(r"[^\S\n]+", " ", re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", " ", s or ""))
    s = re.sub(r"\n\s*\n+", "\n\n", s).strip()
    return s[:limit]


def clean_phone(s: str | None) -> str:
    """A phone number with its country code: "+91 98765 43210" -> "+919876543210".
    The country code is never assumed."""
    digits = re.sub(r"[\s\-().]", "", s or "")
    if not re.fullmatch(r"\+\d{8,15}", digits):
        raise OrderError(400, "Enter your phone number with the country code, for example +91 98765 43210.")
    return digits


def build_order(user: dict, m: dict, detail: dict, specs: dict, *, metal: str, purity: str,
                ring_size: int | None, quantity: int, phone: str, note: str) -> dict:
    """The row for a new order, after checking the choices against what the
    buy page offers for this design."""
    if metal not in m["metals"]:
        raise OrderError(400, "This design isn't made in that metal colour.")
    if purity not in specs["purities_for"].get(metal, []):
        raise OrderError(400, "That gold purity isn't offered in this metal colour.")
    if not 1 <= quantity <= MAX_QTY:
        raise OrderError(400, f"Order between 1 and {MAX_QTY} pieces.")
    size = None
    if m["category"] == "ring":
        if ring_size not in RING_SIZES:
            raise OrderError(400, "Choose your ring size.")
        size = purchase.ring_size(indian=ring_size)
    pur = next(p for p in purchase.PURITIES if p["key"] == purity)
    gold = specs.get("gold")
    snapshot = {
        "design": {"uid": detail["uid"], "design_id": m["design_id"], "folder": m["folders"][0],
                   "category": m["category"], "stone_shape": m.get("stone_shape"),
                   "views": [v["thumb"] for v in detail["views"].get(metal, [])][:4]},
        "selection": {"metal": metal, "purity": purity, "purity_label": pur["label"], "fineness": pur["fineness"],
                      "ring_size": size, "quantity": quantity},
        "gold_weight_g": gold["by_purity"][purity] if gold else None,   # one piece, at the design's own size
        "gold_source": gold["source"] if gold else None,                # "card" / "cad" (an estimate)
        "specs": specs,
    }
    return {
        "user_id": user["uid"], "status": "request", "kind": "as_is",
        "design_uid": detail["uid"], "design_id": m["design_id"], "design_key": purchase.design_key(m),
        "category": m["category"], "metal": metal, "purity": purity,
        "ring_size_in": ring_size if size else None, "quantity": quantity,
        "thumb": detail.get("thumbs_by_metal", {}).get(metal) or detail.get("thumb"),
        "customer_name": user["name"] or user["email"], "customer_email": user["email"],
        "customer_phone": clean_phone(phone), "customer_note": clean_text(note, 1000),
        "snapshot": snapshot,
    }


def build_quote(q: dict, by: str) -> dict:
    """The jeweler's price inputs, checked. Every rate comes from them."""
    if q.get("making_mode") not in MAKING_MODES:
        raise OrderError(400, "Choose how the making charge is counted.")
    num = {}
    for key, lo, hi in (("gold_weight_g", 0.001, 5000), ("gold_rate", 1, 1_000_000), ("making_value", 0, 100_000_000),
                        ("diamond_value", 0, 1_000_000_000), ("other_value", 0, 100_000_000), ("gst_pct", 0, 28)):
        v = q.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
            raise OrderError(400, f"Check the value of {key.replace('_', ' ')}.")
        num[key] = round(float(v), 3 if key == "gold_weight_g" else 2)
    if q["making_mode"] == "percent" and num["making_value"] > 100:
        raise OrderError(400, "A making charge in percent must be 100 or less.")
    return {**num, "making_mode": q["making_mode"],
            "diamond_quality": clean_text(q.get("diamond_quality"), 60),
            "other_label": clean_text(q.get("other_label"), 60),
            "note": clean_text(q.get("note"), 500), "saved_at": _now(), "by": by}


# ---------------------------------------------------------------- money

def _d(x) -> Decimal:
    return Decimal(str(x))


def _money(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.01"), ROUND_HALF_UP))


def bill(q: dict, weight_g: float | None = None) -> dict:
    """The price from the jeweler's inputs. With `weight_g` (weighed at
    quality check) the gold and a per-gram making charge follow the real
    weight; diamonds and other charges stay as quoted. The total is rounded
    to the rupee."""
    w = _d(q["gold_weight_g"] if weight_g is None else weight_g)
    gold = w * _d(q["gold_rate"])
    making = {"per_g": w * _d(q["making_value"]),
              "percent": gold * _d(q["making_value"]) / 100,
              "fixed": _d(q["making_value"])}[q["making_mode"]]
    subtotal = gold + making + _d(q["diamond_value"]) + _d(q["other_value"])
    gst = subtotal * _d(q["gst_pct"]) / 100
    return {"weight_g": float(w), "gold": _money(gold), "making": _money(making),
            "diamonds": _money(_d(q["diamond_value"])), "other": _money(_d(q["other_value"])),
            "subtotal": _money(subtotal), "gst": _money(gst),
            "total": int((subtotal + gst).quantize(Decimal("1"), ROUND_HALF_UP))}


def paid(order: dict) -> float:
    return _money(sum((_d(p["amount"]) for p in order.get("payments") or []), Decimal(0)))


# ---------------------------------------------------------------- stages

def requirement(order: dict, to: str) -> str | None:
    """What must be done before an order may enter stage `to`, or None."""
    if to == "quoted" and not order.get("quote"):
        return "Save the price first."
    if to == "advance_paid" and paid(order) <= 0:
        return "Record the advance payment first."
    if to == "final_bill" and not order.get("actual_weight_g"):
        return "Enter the actual gold weight first."
    return None


def check_move(order: dict, to: str, note: str):
    """Orders move one stage forward or back at a time, or are cancelled with a reason."""
    cur = order["status"]
    if cur == "cancelled":
        raise OrderError(400, "This order is cancelled.")
    if to == "cancelled":
        if cur == "delivered":
            raise OrderError(400, "A delivered order can't be cancelled.")
        if not note:
            raise OrderError(400, "Write why the order is cancelled.")
        return
    if to not in FLOW:
        raise OrderError(400, "Unknown stage.")
    i, j = FLOW.index(cur), FLOW.index(to)
    if j not in (i - 1, i + 1):
        raise OrderError(400, "Orders move one stage at a time.")
    if j == i + 1 and (need := requirement(order, to)):
        raise OrderError(400, need)


def view(order: dict) -> dict:
    """An order as the panel shows it: stage labels, what can be done next, and the money."""
    cur = order["status"]
    q = order.get("quote")
    quote = bill(q) if q else None
    final = bill(q, order["actual_weight_g"]) if q and order.get("actual_weight_g") else None
    got = paid(order)
    owed = final or quote
    nxt = prev = None
    if cur in FLOW:
        i = FLOW.index(cur)
        if i + 1 < len(FLOW):
            nxt = {"key": FLOW[i + 1], "label": LABELS[FLOW[i + 1]], "blocked": requirement(order, FLOW[i + 1])}
        if i > 0:
            prev = {"key": FLOW[i - 1], "label": LABELS[FLOW[i - 1]]}
    return {
        **order, "order_no": order_no(order["id"]), "stage_label": LABELS[cur],
        "totals": {"quote": quote, "final": final, "paid": got,
                   "due": _money(_d(owed["total"]) - _d(got)) if owed else None},
        "next": nxt, "prev": prev,
        "can": {"quote": cur in QUOTE_EDITABLE, "weight": cur in WEIGHT_EDITABLE,
                "pay": cur not in CLOSED, "cancel": cur not in CLOSED},
    }


# ---------------------------------------------------------------- Supabase

async def _req(method: str, path: str, params: dict | None = None, **kw):
    r = await _call(method, path, params=params or {}, **kw)
    if r.status_code == 404:
        raise OrderError(503, "Orders aren't set up yet (run supabase/orders.sql in Supabase).")
    if r.status_code >= 300:
        log.warning("orders: %s %s -> %s %s", method, path, r.status_code, r.text[:300])
        raise OrderError(502, "Orders are unavailable right now. Try again shortly.")
    return r


async def _event(order_id: int, actor: dict, kind: str, *, from_status=None, to_status=None, detail=None,
                 must: bool = False):
    """A line in the order's history. Best effort after a change was saved (a
    lost line must not undo the change); `must` for notes, where the line is
    the whole point."""
    try:
        await _req("POST", EVENTS, json={
            "order_id": order_id, "actor_id": actor["id"], "actor_name": actor["name"], "actor_role": actor["role"],
            "kind": kind, "from_status": from_status, "to_status": to_status, "detail": detail or {},
        }, headers={"Prefer": "return=minimal"})
    except OrderError as e:
        if must:
            raise
        log.warning("order %s: history line %r not saved: %s", order_id, kind, e.message)


async def create(row: dict, actor: dict) -> dict:
    r = await _req("POST", ORDERS, json=row, headers={"Prefer": "return=representation"})
    order = r.json()[0]
    await _event(order["id"], actor, "created", to_status="request",
                 detail={"design_id": row["design_id"], "quantity": row["quantity"]})
    return order


def search_filter(q: str) -> str | None:
    """PostgREST `or=` filter for the panel's search box: order number, design,
    customer name, email or phone. Characters that have a meaning in the
    filter syntax are dropped."""
    q = re.sub(r"[^\w@.+\- ]", "", q or "", flags=re.UNICODE).strip()[:60]
    if not q:
        return None
    parts = [f'{col}.ilike."*{q}*"' for col in ("design_id", "customer_name", "customer_email", "customer_phone")]
    if m := re.fullmatch(r"(?:df-?)?0*(\d{1,12})", q, flags=re.I):
        parts.append(f"id.eq.{m.group(1)}")
    return "(" + ",".join(parts) + ")"


async def list_orders(show: str, q: str, limit: int) -> dict:
    params = {"select": LIST_FIELDS, "order": "created_at.desc", "limit": str(limit)}
    if show == "active":
        params["status"] = "not.in.(delivered,cancelled)"
    elif show in LABELS:
        params["status"] = f"eq.{show}"
    elif show != "all":
        raise OrderError(400, "Unknown list.")
    if f := search_filter(q):
        params["or"] = f
    items = (await _req("GET", ORDERS, params)).json()
    counts = Counter(r["status"] for r in (await _req("GET", ORDERS, {"select": "status", "limit": "100000"})).json())
    return {
        "items": [{**r, "order_no": order_no(r["id"]), "stage_label": LABELS[r["status"]]} for r in items],
        "counts": {"active": sum(n for s, n in counts.items() if s not in CLOSED), "all": sum(counts.values()), **counts},
    }


async def get(order_id: int) -> dict:
    rows = (await _req("GET", ORDERS, {"id": f"eq.{order_id}", "select": "*"})).json()
    if not rows:
        raise OrderError(404, "Order not found.")
    return rows[0]


async def events(order_id: int) -> list[dict]:
    return (await _req("GET", EVENTS, {"order_id": f"eq.{order_id}", "select": "*",
                                       "order": "created_at.desc,id.desc", "limit": "500"})).json()


async def _fresh(order_id: int, seen: str) -> dict:
    """The order, if it is still the version the jeweler was looking at."""
    order = await get(order_id)
    if order["updated_at"] != seen:
        raise OrderError(409, "Someone changed this order a moment ago. It has been reloaded: check it and try again.")
    return order


async def _save(order: dict, changes: dict) -> dict:
    r = await _req("PATCH", ORDERS, {"id": f"eq.{order['id']}", "updated_at": f"eq.{order['updated_at']}"},
                   json={**changes, "updated_at": _now()}, headers={"Prefer": "return=representation"})
    rows = r.json()
    if not rows:   # changed between our read and this write
        raise OrderError(409, "Someone changed this order a moment ago. It has been reloaded: check it and try again.")
    return rows[0]


async def move(order_id: int, to: str, note: str, seen: str, actor: dict) -> dict:
    order = await _fresh(order_id, seen)
    note = clean_text(note, 500)
    check_move(order, to, note)
    changes = {"status": to}
    if to == "cancelled":
        changes["cancel_reason"] = note
    saved = await _save(order, changes)
    await _event(order_id, actor, "cancelled" if to == "cancelled" else "stage",
                 from_status=order["status"], to_status=to, detail={"note": note} if note else {})
    return saved


async def save_quote(order_id: int, quote: dict, seen: str, actor: dict) -> dict:
    order = await _fresh(order_id, seen)
    if order["status"] not in QUOTE_EDITABLE:
        raise OrderError(400, "The price is locked once the customer has approved it. "
                              "Move the order back to \"Price sent\" to change it.")
    q = build_quote(quote, actor["name"])
    total = bill(q)["total"]
    saved = await _save(order, {"quote": q, "quoted_total": total})
    await _event(order_id, actor, "quote", detail={"total": total, "gold_rate": q["gold_rate"],
                                                    "gold_weight_g": q["gold_weight_g"]})
    return saved


async def add_payment(order_id: int, amount: float, method: str, ref: str, seen: str, actor: dict) -> dict:
    order = await _fresh(order_id, seen)
    if order["status"] in CLOSED:
        raise OrderError(400, "Payments can't be added to a closed order.")
    if method not in PAY_METHODS:
        raise OrderError(400, "Choose how it was paid.")
    if not 0 < amount <= 1_000_000_000:
        raise OrderError(400, "Enter the amount received.")
    pay = {"id": secrets.token_hex(4), "amount": round(float(amount), 2), "method": method,
           "ref": clean_text(ref, 80), "at": _now(), "by": actor["name"]}
    saved = await _save(order, {"payments": [*(order.get("payments") or []), pay]})
    await _event(order_id, actor, "payment", detail={"amount": pay["amount"], "method": method, "ref": pay["ref"]})
    return saved


async def remove_payment(order_id: int, pay_id: str, seen: str, actor: dict) -> dict:
    order = await _fresh(order_id, seen)
    if order["status"] in CLOSED:
        raise OrderError(400, "Payments of a closed order can't be changed.")
    pays = order.get("payments") or []
    gone = next((p for p in pays if p["id"] == pay_id), None)
    if not gone:
        raise OrderError(404, "Payment not found.")
    saved = await _save(order, {"payments": [p for p in pays if p["id"] != pay_id]})
    await _event(order_id, actor, "payment_removed", detail={"amount": gone["amount"], "method": gone["method"]})
    return saved


async def set_weight(order_id: int, grams: float, note: str, seen: str, actor: dict) -> dict:
    order = await _fresh(order_id, seen)
    if order["status"] not in WEIGHT_EDITABLE:
        raise OrderError(400, "The actual weight is entered at the quality check stage.")
    if not 0.001 <= grams <= 5000:
        raise OrderError(400, "Enter the weight in grams.")
    saved = await _save(order, {"actual_weight_g": round(float(grams), 3)})
    await _event(order_id, actor, "weight", detail={"grams": round(float(grams), 3), "note": clean_text(note, 300)})
    return saved


async def add_note(order_id: int, text: str, actor: dict):
    await get(order_id)
    text = clean_text(text, 1000)
    if not text:
        raise OrderError(400, "Write a note first.")
    await _event(order_id, actor, "note", detail={"text": text}, must=True)
