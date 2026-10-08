"""An in-memory stand-in for the few Supabase REST (PostgREST) features the
orders module uses, so the order flow can be tested without a database.

Supports GET / POST / PATCH on /rest/v1/<table> with `select`, `order`,
`limit`, `eq.`, `in.`, `not.in.` filters and an `or=(... .ilike. / .eq. ...)`
group, and the `Prefer: return=representation` header.
"""
import itertools
import json
import re
import time
from datetime import datetime, timezone

import httpx

DEFAULTS = {
    "orders": {"status": "request", "kind": "as_is", "quote": None, "quoted_total": None, "payments": [],
               "actual_weight_g": None, "cancel_reason": None, "customer_note": ""},
    "order_events": {"detail": {}, "from_status": None, "to_status": None, "actor_name": ""},
}


def _stamp() -> str:
    time.sleep(0.000002)   # distinct, ordered timestamps like Postgres now()
    return datetime.now(timezone.utc).isoformat()


class FakeRest:
    def __init__(self, tables=("orders", "order_events")):
        self.tables = {t: [] for t in tables}
        self.ids = {t: itertools.count(1) for t in tables}
        self.calls = []

    # the same signature as jewelsearch.auth._call
    async def __call__(self, method, path, headers=None, params=None, json=None, **_):
        self.calls.append((method, path, dict(params or {})))
        table = path.removeprefix("/rest/v1/")
        if table not in self.tables:
            return httpx.Response(404, json={"message": f"relation {table} does not exist"})
        rows = self.tables[table]
        params = dict(params or {})
        want = "return=representation" in (headers or {}).get("Prefer", "")
        if method == "POST":
            new = []
            for body in (json if isinstance(json, list) else [json]):
                now = _stamp()
                row = {**DEFAULTS.get(table, {}), **_copy(body), "id": next(self.ids[table]), "created_at": now}
                if table == "orders":
                    row["updated_at"] = now
                rows.append(row)
                new.append(row)
            return httpx.Response(201, json=_copy(new) if want else None)
        hits = [r for r in rows if _matches(r, params)]
        if method == "PATCH":
            for r in hits:
                r.update(_copy(json))
            return httpx.Response(200, json=_copy(hits) if want else None)
        if method == "GET":
            for key, desc in reversed([(k.split(".")[0], k.endswith(".desc"))
                                       for k in params.get("order", "").split(",") if k]):
                hits.sort(key=lambda r: (r.get(key) is None, r.get(key)), reverse=desc)
            hits = hits[:int(params.get("limit", 10 ** 9))]
            cols = params.get("select", "*")
            if cols != "*":
                names = [c.split(":")[-1] for c in cols.split(",")]
                hits = [{c: r.get(c) for c in names} for r in hits]
            return httpx.Response(200, json=_copy(hits))
        return httpx.Response(405, json={"message": "method not supported by the fake"})


def _copy(x):
    return json.loads(json.dumps(x))


def _test(value, expr: str) -> bool:
    if expr.startswith("eq."):
        return str(value) == expr[3:].strip('"')
    if expr.startswith("in.("):
        return str(value) in expr[4:-1].split(",")
    if expr.startswith("not.in.("):
        return str(value) not in expr[8:-1].split(",")
    if expr.startswith("ilike."):
        pat = re.escape(expr[6:].strip('"')).replace(r"\*", ".*")
        return value is not None and re.fullmatch(pat, str(value), re.I | re.S) is not None
    raise ValueError(f"filter not supported by the fake: {expr}")


def _matches(row: dict, params: dict) -> bool:
    for key, expr in params.items():
        if key in ("select", "order", "limit"):
            continue
        if key == "or":
            terms = re.findall(r'(\w+)\.((?:ilike|eq)\.(?:"[^"]*"|[^,)]*))', expr)
            if not any(_test(row.get(col), e) for col, e in terms):
                return False
        elif not _test(row.get(key), expr):
            return False
    return True
