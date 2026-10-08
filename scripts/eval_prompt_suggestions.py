"""Do the suggested prompts (jewelsearch/recommend.py) anticipate what a shopper searches next?

There is no real multi-user history to test with, so this simulates shoppers:
each has favourite types, maybe a favourite metal, a few looks and a language,
and makes 8 searches over a month. A next search repeats an earlier one (35%),
tries a new look in a type they like (45%) or another type (20%). The 5
suggestions are made from the first 7 searches (plus everyone else's searches
as "popular"), and the 8th is held out.

A hit: one of the 5 suggestions means the held-out search: the same type, its
metal if it has one, and at least one of its looks if it has any.

Compared with: the fixed examples the page showed before, the shopper's own
last 5 searches, and the most popular searches. Simulated shoppers follow the
same kind of taste model the recommender assumes, so read the numbers as a
sanity check, not a field result.

    .venv/bin/python scripts/eval_prompt_suggestions.py [--set MAX_RECENT=3 ...]
"""
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jewelsearch.query import parse  # noqa: E402
from jewelsearch.recommend import Recommender, Spec  # noqa: E402
from jewelsearch.search import SearchEngine  # noqa: E402

LOOKS = {
    "ring": ["solitaire", "thin", "wide", "halo", "floral", "heart", "minimal", "statement", "bridal", "everyday",
             "engagement"],
    "earrings": ["stud", "hoop", "drop", "floral", "minimal", "statement", "bridal", "everyday"],
    "pendant": ["heart", "solitaire", "floral", "minimal", "bridal", "everyday"],
    "necklace": ["choker", "bridal", "statement", "minimal", "everyday"],
    "bracelet": ["bangle", "cuff", "minimal", "statement", "everyday"],
}
TYPE_P = {"ring": .5, "earrings": .25, "pendant": .1, "necklace": .08, "bracelet": .07}
OLD_EXAMPLES = ["A rose gold ring for my mother, wearable every day", "મારી બહેન માટે નાની અને સાદી કાનની બુટ્ટી",
                "शादी के लिए भारी हीरे का हार", "floral pendant in yellow gold as a gift", "gents ring bina stone"]
N_USERS, N_SEARCHES = 80, 8


def persona(rng):
    types = list(TYPE_P)
    main = rng.choices(types, [TYPE_P[t] for t in types])[0]
    second = rng.choice([t for t in types if t != main])
    return {"types": [main, main, main, second], "metal": rng.choice(["rose_gold", "white_gold", "yellow_gold", None, None]),
            "looks": {t: rng.sample(LOOKS[t], 2) for t in types},
            "lang": rng.choices(["en", "gu", "hi"], [.6, .25, .15])[0]}


def make_search(rng, rec, p, history):
    for _ in range(50):
        r = rng.random()
        if history and r < .35:
            text, spec, _ = rng.choice(history)
            return text, spec
        t = rng.choice(p["types"]) if r < .8 else rng.choice(list(TYPE_P))
        looks = p["looks"][t] if r < .8 else LOOKS[t]
        spec = Spec(t, p["metal"] if rng.random() < .7 else None, tuple(rng.sample(looks, rng.choice([1, 1, 2]))))
        c = rec._make(spec, p["lang"], "x", "")
        if c:
            return (c.text, spec)
    raise RuntimeError("no search could be made")


def same(a: Spec, b: Spec) -> bool:
    return a.cat == b.cat and a.metal == b.metal and set(a.intents) == set(b.intents)


def means(text, spec: Spec) -> bool:
    q = parse(text)
    return (spec.cat in q.categories and (spec.metal is None or q.metal == spec.metal)
            and (not spec.intents or bool(set(spec.intents) & set(q.intents))))


def main():
    # try a setting: --set MAX_RECENT=3
    from jewelsearch import recommend
    args = sys.argv[1:]
    for k, v in (a.split("=", 1) for a in (args[args.index("--set") + 1:] if "--set" in args else [])):
        setattr(recommend, k, type(getattr(recommend, k))(v))
        print(f"setting {k} = {v}")
    rng = random.Random(7)
    e = SearchEngine(judge=lambda t: "yes")
    rec = Recommender(e)
    now = datetime.now(timezone.utc)
    users = []
    for u in range(N_USERS):
        p = persona(rng)
        searches, days = [], sorted(rng.sample(range(30), N_SEARCHES), reverse=True)
        for d in days:
            searches.append((*make_search(rng, rec, p, searches), now - timedelta(days=d)))
        users.append((f"u{u}", p, searches))
    # everyone's searches, meaning only, as the server reads them
    popular = [{"user_id": uid, "category": s.cat, "metal": s.metal, "intents": list(s.intents), "shape": None,
                "refused": None, "created_at": t.isoformat()} for uid, _, ss in users for _, s, t in ss[:-1]]
    names = ("recommend", "old fixed examples", "own last 5", "popular only")
    hits = {k: {"repeat": 0, "new": 0} for k in names}
    n_kind = {"repeat": 0, "new": 0}
    types_shown, own_script, n_items, t_total = 0, 0, 0, 0.0
    pop_rec = Recommender(e)
    pop_list = [it["text"] for it in pop_rec([], [], popular, "nobody")["items"]]
    for uid, p, ss in users:
        past, (target_text, target, _) = ss[:-1], ss[-1]
        rows = [{"id": i, "query": text, "understood": {}, "created_at": t.isoformat()}
                for i, (text, _, t) in enumerate(reversed(past))]
        t0 = time.perf_counter()
        res = rec(rows, [], popular, uid)
        t_total += time.perf_counter() - t0
        items = [it["text"] for it in res["items"]]
        own = list(dict.fromkeys(r["query"] for r in rows))[:5]
        # a repeat: the held-out search means the same as an earlier one; new: it doesn't
        kind = "repeat" if any(same(s, target) for _, s, _ in past) else "new"
        n_kind[kind] += 1
        for name, lst in (("recommend", items), ("old fixed examples", OLD_EXAMPLES), ("own last 5", own),
                          ("popular only", pop_list)):
            hits[name][kind] += any(means(x, target) for x in lst)
        types_shown += len({(parse(x).categories or ["?"])[0] for x in items})
        from jewelsearch.suggest import script
        own_script += sum(script(x) == p["lang"] for x in items)
        n_items += len(items)
    print(f"{N_USERS} simulated shoppers, {N_SEARCHES} searches each, the last one held out\n")
    print(f"next search among the 5 suggestions ({n_kind['repeat']} repeats, {n_kind['new']} new searches):")
    print(f"  {'':20s} {'all':>7s} {'repeat':>8s} {'new':>7s}")
    for name, h in hits.items():
        print(f"  {name:20s} {(h['repeat'] + h['new']) / N_USERS:7.1%} {h['repeat'] / max(1, n_kind['repeat']):8.1%}"
              f" {h['new'] / max(1, n_kind['new']):7.1%}")
    print(f"\ndifferent types among the 5:      {types_shown / N_USERS:.1f}")
    print(f"suggestions in the shopper's script: {own_script / max(n_items, 1):.0%}")
    print(f"time per shopper:                  {t_total / N_USERS * 1000:.0f} ms")


if __name__ == "__main__":
    main()
