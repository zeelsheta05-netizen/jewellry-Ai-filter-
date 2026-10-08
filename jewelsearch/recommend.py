"""The 5 suggested prompts under the search bar, chosen for each shopper.

Signals
- The shopper's own searches (search_history): type, metal, styles, stone cut
  and language, each search weighted by how recent it is (half-life 14 days)
  and repeats counted. Searches that were refused (not jewellery) are ignored.
- Their favourites: each liked design's type and look (its attribute tags) and
  its picture.
- What everyone searched lately ("popular"): only the parsed meaning of other
  people's searches (type, metal, style), never the words they typed, and only
  when at least TRENDING_MIN_USERS other people searched it.
- The collection: types and looks it has many designs of, for discovery.

Candidates (all written in the shopper's script: English, ગુજરાતી or हिन्दी)
- recent: the shopper's own frequent / latest searches, to run again
- for_you: their favourite type + metal + styles, a style they like in another
  type, and the next refinement of what they search (from jewelsearch/suggest.py)
- popular: what other shoppers search
- explore: well stocked types and looks of the collection

Every generated prompt is parsed back and must mean exactly what it was made
from, and must fill a page of designs without relaxing a filter (the rule
suggest.py follows); the shopper's own past prompts must still pass the
jewellery check (domain.py).

Ranking: each candidate is embedded with the image model's text encoder (the
same space as the designs' pictures) and scored on
  affinity   how much of the shopper's profile it matches (type, metal, styles)
  taste      similarity to the shopper's own searches (recency weighted)
  favourites how strongly it picks out the designs they saved, among all designs
  novelty    not a near repeat of a search they just made
  popularity how many other shoppers search it
  depth      how many designs it finds
then 5 are picked: at most 2 to run again, at least 1 for discovery when there
is history, and the rest greedily by score minus similarity to those already
picked (MMR), at most 3 of one type. A new shopper gets popular and explore
prompts across all types.
"""
import math
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import CATEGORIES, METALS
from .query import parse
from .search import TOP_K
from .suggest import (CUT_WORDS, REFINEMENTS, TYPE_WORDS, VOCAB, Suggester, _place, script)

N_SHOW = 5
MAX_RECENT = 2            # own searches to run again...
MAX_RECENT_REPEATER = 3   # ...or for a shopper who often repeats searches (REPEATER_RATE of them or more)
REPEATER_RATE = 0.4
MAX_DISCOVERY = {"popular": 1, "explore": 1}   # per shopper with history (a new shopper gets only these)
MAX_PER_TYPE = 3
HALF_LIFE_DAYS = 14.0     # a search counts half as much after two weeks
FAV_WEIGHT = 1.5          # a saved design counts like 1.5 searches
TRENDING_DAYS = 30
TRENDING_HALF_LIFE = 7.0
TRENDING_MIN_USERS = 2    # other people who searched it before it is shown as popular
MMR_LAMBDA = 0.7          # 1 = score only, lower = more varied
W = {"affinity": 1.2, "session": 0.8, "taste": 1.0, "favourites": 1.5, "novelty": 0.8, "popularity": 0.8, "depth": 0.3}
MAX_OTHER_TYPE = 1        # "your style, in <a type you haven't searched>" chips
# who a search is for or why: says nothing about the look, so not part of the taste profile
NOT_STYLE = {"women", "gift", "kids", "diamond"}
# a saved design's look (attribute tags of the index) as the parser's intents
TAG_INTENT = {("stones", "solitaire"): "solitaire", ("stones", "pave"): "cluster", ("band", "thin"): "thin",
              ("band", "wide"): "wide", ("weight", "delicate"): "minimal", ("weight", "statement"): "statement",
              ("form", "stud"): "stud", ("form", "hoop"): "hoop", ("form", "drop"): "drop",
              ("form", "bangle"): "bangle", ("form", "cuff"): "cuff", ("form", "choker"): "choker"}
NOUN_EN = {"ring": "rings", "earrings": "earrings", "pendant": "pendants", "necklace": "necklaces",
           "bracelet": "bracelets"}
_POST = re.compile(r"^(for|with|without) |માટે$|સાથે$|વગર$|के लिए$|के साथ$|के बिना$")


def _intent_phrases() -> dict:
    """lang -> intent -> (where, words): every refinement word and every
    completion word that means one look and nothing else, in each script."""
    out = {lang: {} for lang in VOCAB}
    for _, claim, words, _ in REFINEMENTS:
        if claim.startswith("intent:"):
            for lang, wf in words.items():
                out[lang].setdefault(claim[7:], wf)
    for lang, phrases in VOCAB.items():
        for p in phrases:
            q = parse(p)
            if len(q.intents) == 1 and not (q.categories or q.metal or q.shape or q.not_intents):
                out[lang].setdefault(q.intents[0], ("post" if _POST.search(p) else "pre", p))
    return out


INTENT_WORDS = _intent_phrases()
METAL_WORDS = {lang: {claim[6:]: wf[lang] for _, claim, wf, _ in REFINEMENTS if claim.startswith("metal:") and lang in wf}
               for lang in VOCAB}


@dataclass(frozen=True)
class Spec:
    """What a suggested prompt means: the search's own reading of it."""
    cat: str
    metal: str | None = None
    intents: tuple = ()
    shape: str | None = None


@dataclass
class Cand:
    text: str
    kind: str                  # recent | for_you | popular | explore
    why: str
    spec: Spec | None = None
    count: int = 0
    score: float = 0.0
    extra: dict = field(default_factory=dict)


def _age_days(iso: str, now: float) -> float:
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 365.0
    return max(0.0, (now - t) / 86400)


def _decay(days: float, half_life: float) -> float:
    return 0.5 ** (days / half_life)


def _label(spec: Spec) -> str:
    """An English description of a profile, for the "why" line."""
    words = [spec.metal.replace("_", " ")] if spec.metal else []
    return " ".join(words + [NOUN_EN[spec.cat]])


class Recommender:
    def __init__(self, engine, suggester: Suggester | None = None):
        self.engine = engine
        self.suggester = suggester or Suggester(engine)
        self._vec: OrderedDict[str, np.ndarray] = OrderedDict()   # text -> its vector (bounded cache)
        self._vec_lock = threading.Lock()
        self._explore: dict[str, list[Cand]] = {}
        self._render_cache: dict = {}

    # ---- public ------------------------------------------------------------
    def __call__(self, history: list[dict], favorites: list[dict], trending: list[dict],
                 user_id: str | None = None, now: float | None = None) -> dict:
        now = now or time.time()
        searches = self._own_searches(history, now)
        favs = self._favourites(favorites, now)
        prof = self._profile(searches, favs)
        lang = prof["lang"]
        cands = (self._recent(searches) + self._for_you(prof, lang) + self._popular(trending, user_id, now, lang)
                 + self._explore_pool(lang))
        picked = self._pick(cands, prof, searches, favs)
        return {"items": [{"text": c.text, "kind": c.kind, "why": c.why, "count": c.count or None} for c in picked],
                "personal": bool(searches or favs), "lang": lang}

    # ---- signals -------------------------------------------------------------
    def _own_searches(self, rows: list[dict], now: float) -> list[dict]:
        out = []
        for r in rows:
            u = r.get("understood") or {}
            text = " ".join((r.get("query") or "").split())
            if not text or u.get("refused"):
                continue
            q = parse(text)
            cats = list(q.categories)
            if r.get("category_override") in CATEGORIES:
                cats = [r["category_override"]]
            metal = r["metal_override"] if r.get("metal_override") in METALS else q.metal
            out.append({"text": text, "q": q, "cats": cats, "metal": metal,
                        "w": _decay(_age_days(r.get("created_at", ""), now), HALF_LIFE_DAYS)})
        return out

    def _favourites(self, rows: list[dict], now: float) -> list[dict]:
        out = []
        for r in rows:
            uid = r.get("design_uid")
            m = self.engine.by_uid.get(uid)
            if m is None or m["design_id"] != r.get("design_id"):   # index rebuilt: find it by its id
                uid = next((u for u, mm in self.engine.by_uid.items() if mm["design_id"] == r.get("design_id")), None)
                m = self.engine.by_uid.get(uid)
            if m is None:
                continue
            looks = [i for key, i in TAG_INTENT.items()
                     if key in self.engine.attr and np.nan_to_num(self.engine.attr[key][uid]) >= 0.5]
            out.append({"uid": uid, "cat": m["category"], "looks": looks,
                        "w": FAV_WEIGHT * _decay(_age_days(r.get("created_at", ""), now), HALF_LIFE_DAYS)})
        return out

    def _profile(self, searches: list[dict], favs: list[dict]) -> dict:
        cat, metal, look, shape, lang = Counter(), Counter(), Counter(), Counter(), Counter()
        for s in searches:
            w = s["w"]
            for c in s["cats"]:
                cat[c] += w / len(s["cats"])
            if s["metal"]:
                metal[s["metal"]] += w
            for i in s["q"].intents:
                if i not in NOT_STYLE:
                    look[i] += w
            if s["q"].shape:
                shape[s["q"].shape] += w
            lang[script(s["text"])] += w
        for f in favs:
            cat[f["cat"]] += f["w"]
            for i in f["looks"]:
                look[i] += f["w"] / max(1, len(f["looks"]))

        def share(c):
            total = sum(c.values())
            return {k: v / total for k, v in c.most_common()} if total else {}
        texts = [s["text"] for s in searches]
        taste = None
        if texts:
            v = self._embed(texts)
            taste = (np.array([s["w"] for s in searches])[:, None] * v).sum(0)
            taste /= np.linalg.norm(taste) + 1e-9
        latest = searches[0] if searches else None
        return {"cat": share(cat), "metal": share(metal), "look": share(look), "shape": share(shape),
                "now": latest and {"cats": latest["cats"], "metal": latest["metal"],
                                   "looks": [i for i in latest["q"].intents if i not in NOT_STYLE]},
                "lang": max(lang, key=lang.get) if lang else "en", "taste": taste,
                "weight": sum(s["w"] for s in searches) + sum(f["w"] for f in favs)}

    # ---- candidates ------------------------------------------------------------
    def _recent(self, searches: list[dict]) -> list[Cand]:
        """Own searches to run again: near-duplicates merged by meaning, scored by repeats and recency."""
        groups: dict = {}
        for i, s in enumerate(searches):
            q = s["q"]
            key = (tuple(sorted(s["cats"])), s["metal"], frozenset(q.intents), q.shape, frozenset(q.not_intents),
                   q.free_text())
            g = groups.setdefault(key, {"text": s["text"], "first": i, "n": 0, "w": 0.0})
            g["n"] += 1
            g["w"] += s["w"]
        # the latest search (what they are after now) first, then the most frequent ones
        ordered = sorted(groups.values(), key=lambda g: (g["first"] != 0, -g["w"]))
        out = []
        for rank, g in enumerate(ordered[:MAX_RECENT + 2]):
            if not self.engine.domain.check(g["text"]).ok:
                continue   # saved before the jewellery check existed, or not jewellery after all
            why = f"You searched this {g['n']} times" if g["n"] > 1 else "Your latest search" if g["first"] == 0 \
                else "Your recent search"
            out.append(Cand(g["text"], "recent", why, extra={"w": 10 - rank, "n": g["n"]}))
        return out

    def _for_you(self, prof: dict, lang: str) -> list[Cand]:
        if not prof["cat"]:
            return []
        cats = [c for c, s in prof["cat"].items() if s >= 0.15][:2]
        metal = next((m for m, s in prof["metal"].items() if s >= 0.35), None)
        looks = [i for i, s in prof["look"].items() if s >= 0.1][:4]
        shape = next((s for s, v in prof["shape"].items() if v >= 0.3), None)
        specs: list[tuple[Spec, str]] = []
        for c in cats:
            base = Spec(c, metal)
            specs.append((base, f"Because you look at {_label(base)}"))
            for i in looks:
                specs.append((Spec(c, metal, (i,)), f"Your style, in {NOUN_EN[c]}"))
            for i, j in zip(looks, looks[1:]):
                specs.append((Spec(c, metal, (i, j)), f"Your style, in {NOUN_EN[c]}"))
            if shape:
                specs.append((Spec(c, metal, (), shape), f"You like {shape} stones"))
        # what they are looking at now: the latest search's type with the looks they like
        now = prof["now"]
        if now and now["cats"]:
            c = now["cats"][0]
            for i in dict.fromkeys(now["looks"] + looks):
                specs.append((Spec(c, now["metal"] or metal, (i,)), f"More {NOUN_EN[c]} in your style"))
                specs.append((Spec(c, None, (i,)), f"More {NOUN_EN[c]} in your style"))
        # a look they like, in a type they haven't searched ("complete the look")
        for c in CATEGORIES:
            if c not in prof["cat"] and looks:
                specs.append((Spec(c, metal, (looks[0],)), f"Your style, in {NOUN_EN[c]}"))
        out = [c for spec, why in specs if (c := self._make(spec, lang, "for_you", why))]
        for c in out:
            c.extra["other_type"] = c.spec.cat not in prof["cat"]
        # the next step from what they search most: suggest.py's checked refinements
        top = self._make(Spec(cats[0], metal), lang, "for_you", "") if cats else None
        if top:
            for it in self.suggester(top.text + " ")["items"]:
                q = parse(it["text"])
                if it["kind"] == "refine" and not set(q.intents) <= NOT_STYLE | set(top.spec.intents):
                    out.append(Cand(it["text"], "for_you", f"Something new in {NOUN_EN[cats[0]]}",
                                    Spec(cats[0], q.metal, tuple(q.intents), q.shape), it["count"] or self._count(q)))
        return out

    def _popular(self, rows: list[dict], user_id: str | None, now: float, lang: str) -> list[Cand]:
        """What other shoppers search, by meaning only, when enough of them did."""
        users, score = {}, Counter()
        for r in rows:
            if r.get("refused") or r.get("user_id") == user_id or r.get("category") not in CATEGORIES:
                continue
            looks = tuple(sorted(i for i in (r.get("intents") or []) if i not in NOT_STYLE)[:2])
            spec = Spec(r["category"], r.get("metal") if r.get("metal") in METALS else None, looks, r.get("shape"))
            users.setdefault(spec, set()).add(r.get("user_id"))
            score[spec] += _decay(_age_days(r.get("created_at", ""), now), TRENDING_HALF_LIFE)
        out = []
        for spec, s in score.most_common(30):
            if len(users[spec]) >= TRENDING_MIN_USERS:
                c = self._make(spec, lang, "popular", "Popular with shoppers")
                if c:
                    c.extra["popularity"] = math.log1p(s)
                    out.append(c)
        return out

    def _explore_pool(self, lang: str) -> list[Cand]:
        """Well stocked types and looks of the collection (the same for everyone; cached)."""
        if lang not in self._explore:
            pool = []
            for c in CATEGORIES:
                base = self._make(Spec(c), lang, "explore", f"Explore {NOUN_EN[c]}")
                if not base:
                    continue
                looks = []
                for it in self.suggester(base.text + " ")["items"]:
                    q = parse(it["text"])
                    if it["kind"] == "refine" and not q.metal and not set(q.intents) <= NOT_STYLE:
                        looks.append(Cand(it["text"], "explore", f"Explore {NOUN_EN[c]}",
                                          Spec(c, None, tuple(q.intents), q.shape), 0 if q.shape else it["count"] or self._count(q)))
                pool += looks or [base]   # "solitaire rings" says more than "rings"
            self._explore[lang] = pool
        return [Cand(c.text, c.kind, c.why, c.spec, c.count) for c in self._explore[lang]]

    # ---- writing and checking a prompt ---------------------------------------------
    def _render(self, spec: Spec, lang: str) -> str | None:
        """The prompt for a meaning, in a script: "rose gold solitaire rings",
        "રોઝ ગોલ્ડ સોલિટેર વીંટી", "सॉलिटेयर अंगूठी पुरुषों के लिए"."""
        noun = TYPE_WORDS[lang][spec.cat]
        steps = [INTENT_WORDS[lang].get(i) for i in spec.intents]
        if spec.shape:
            steps.append(("pre", CUT_WORDS[lang].get(spec.shape)))
        metal = [METAL_WORDS[lang].get(spec.metal)] if spec.metal else []
        # English puts the metal first ("rose gold heart pendants"); Gujarati and Hindi put
        # words right before the piece, so the metal goes in first ("યલો ગોલ્ડ દિલ પેન્ડન્ટ")
        steps = steps + metal if lang == "en" else metal + steps
        text = noun[1] if isinstance(noun, tuple) else noun
        for wf in steps:
            if not wf or not wf[1]:
                return None
            text = _place(text, wf[0], wf[1], lang)
        return text

    def _count(self, q) -> int:
        relaxed = []
        mask, _ = self.engine._filters(q, relaxed)
        n = int(mask.sum())
        if relaxed or n < TOP_K:
            return 0
        if q.shape and int((mask & (self.engine.shape == q.shape)).sum()) < TOP_K:
            return 0
        return n

    def _make(self, spec: Spec, lang: str, kind: str, why: str) -> Cand | None:
        key = (spec, lang)
        if key not in self._render_cache:
            text, n = self._render(spec, lang), 0
            if text:
                q = parse(text)
                means_it = (q.categories == [spec.cat] and q.metal == spec.metal and q.shape == spec.shape
                            and set(spec.intents) <= set(q.intents) and not q.not_intents and not q.free_text())
                n = self._count(q) if means_it else 0
            self._render_cache[key] = (text, n) if n else None
        hit = self._render_cache[key]
        # with a stone cut the count is every design of the type (the cut only reorders them): not shown
        return Cand(hit[0], kind, why, spec, 0 if spec.shape else hit[1]) if hit else None

    # ---- vectors -----------------------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._vec_lock:
            todo = [t for t in dict.fromkeys(texts) if t not in self._vec]
        if todo:
            vs = self.engine.emb.texts([f"a photo of {t}" for t in todo])
            with self._vec_lock:
                for t, v in zip(todo, vs):
                    self._vec[t] = v
                while len(self._vec) > 20000:
                    self._vec.popitem(last=False)
        with self._vec_lock:
            return np.stack([self._vec[t] for t in texts])

    # ---- ranking -------------------------------------------------------------------
    def _pick(self, cands: list[Cand], prof: dict, searches: list[dict], favs: list[dict]) -> list[Cand]:
        seen, uniq = set(), []
        for c in cands:   # one per wording, the earliest (most personal) source wins
            k = c.text.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        if not uniq:
            return []
        vecs = self._embed([c.text for c in uniq])
        recent_vecs = self._embed([s["text"] for s in searches[:5]]) if searches else None
        fav_uids = [f["uid"] for f in favs]
        n_designs = len(self.engine.meta)
        top_pop = max((c.extra.get("popularity", 0.0) for c in uniq), default=0.0) or 1.0
        for c, v in zip(uniq, vecs):
            f = {}
            spec = c.spec
            if spec and prof["now"]:
                f["session"] = float(spec.cat in prof["now"]["cats"])   # the type they are looking at now
            if spec:
                f["affinity"] = (prof["cat"].get(spec.cat, 0) + (prof["metal"].get(spec.metal, 0) if spec.metal else 0)
                                 + sum(prof["look"].get(i, 0) for i in spec.intents) / max(1, len(spec.intents))
                                 + (prof["shape"].get(spec.shape, 0) if spec.shape else 0))
            f["taste"] = float(v @ prof["taste"]) if prof["taste"] is not None else 0.0
            if fav_uids:
                # where the saved designs rank among all designs for this prompt (0.5 = no better than chance)
                sims = self.engine.front @ v
                ranks = (sims[None, :] < sims[fav_uids][:, None]).mean(1)
                f["favourites"] = float(ranks.mean()) - 0.5
            if recent_vecs is not None and c.kind != "recent":
                f["novelty"] = -max(0.0, float((recent_vecs @ v).max()) - 0.9) * 10   # only near repeats lose
            f["popularity"] = c.extra.get("popularity", 0.0) / top_pop   # 0..1: a tie-breaker, never the main reason
            f["depth"] = math.log1p(c.count) / math.log1p(n_designs) if c.count else 0.0
            c.score = sum(W[k] * val for k, val in f.items())
            if c.kind == "recent":
                c.score = 10 + c.extra["w"]   # always considered first (latest, then most frequent), within MAX_RECENT
        order = sorted(range(len(uniq)), key=lambda i: -uniq[i].score)
        picked: list[int] = []
        per_type = Counter()

        def type_of(c):
            return c.spec.cat if c.spec else (parse(c.text).categories or ["any"])[0]

        personal = bool(searches or favs)
        per_type_max = MAX_PER_TYPE if personal else 1   # a new shopper sees every type once
        # how many own searches to offer again follows the shopper's habit
        meanings = [(tuple(sorted(s["cats"])), s["metal"], frozenset(set(s["q"].intents) - NOT_STYLE), s["q"].shape)
                    for s in searches]
        repeats = 1 - len(set(meanings)) / len(meanings) if len(meanings) >= 4 else 0.0
        max_recent = MAX_RECENT_REPEATER if repeats >= REPEATER_RATE else MAX_RECENT

        def can_take(i):
            c = uniq[i]
            if c.kind == "recent" and sum(uniq[j].kind == "recent" for j in picked) >= max_recent:
                return False
            if personal and c.kind in MAX_DISCOVERY and sum(uniq[j].kind == c.kind for j in picked) >= MAX_DISCOVERY[c.kind]:
                return False
            if meaning(c) in shown:
                return False
            if c.extra.get("other_type") and sum(bool(uniq[j].extra.get("other_type")) for j in picked) >= MAX_OTHER_TYPE:
                return False
            return per_type[type_of(c)] < per_type_max

        def meaning(c):
            """What a chip means, in any script: two chips never mean the same (the text encoder
            reads Gujarati too poorly to see that "ફૂલ બુટ્ટી" repeats "ફૂલવાળી કાનની બુટ્ટી")."""
            if c.spec:
                return (c.spec.cat, c.spec.metal, frozenset(set(c.spec.intents) - NOT_STYLE), c.spec.shape)
            q = parse(c.text)
            return ((q.categories or ["any"])[0], q.metal, frozenset(set(q.intents) - NOT_STYLE), q.shape)

        shown = set()

        def take(i):
            picked.append(i)
            per_type[type_of(uniq[i])] += 1
            shown.add(meaning(uniq[i]))

        for i in order:   # own searches first
            if uniq[i].kind == "recent" and can_take(i):
                take(i)
        while len(picked) < N_SHOW:
            need_discovery = (personal and len(picked) == N_SHOW - 1
                              and not any(uniq[j].kind in ("popular", "explore") for j in picked))
            best, best_val = None, -1e9
            for i in order:
                if i in picked or not can_take(i):
                    continue
                if need_discovery and uniq[i].kind not in ("popular", "explore"):
                    continue
                red = max((float(vecs[i] @ vecs[j]) for j in picked), default=0.0)
                val = MMR_LAMBDA * uniq[i].score - (1 - MMR_LAMBDA) * 3 * red
                if val > best_val:
                    best, best_val = i, val
            if best is None:
                if need_discovery:   # nothing for discovery: fill with anything
                    need_discovery = False
                    personal = False
                    continue
                break
            take(best)
        return [uniq[i] for i in picked]
