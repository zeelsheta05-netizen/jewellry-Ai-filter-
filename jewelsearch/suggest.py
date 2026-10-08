"""Search-box suggestions while the shopper types, like a web search engine's
autocomplete, but made from this collection instead of other people's searches.

The page shows, under the box:
  1. the shopper's own past searches that match what is typed (the page
     filters the history it already loaded from the server; another user's
     prompts are never shown, since they can name family members or occasions)
  2. completions of the word being typed, from words the search understands
     ("bold ri" -> "bold rings"), and
  3. refinements: the prompt plus one more thing (type, wearer, metal,
     stones, form, occasion, stone cut), written in the shopper's script.

Every suggestion is checked before it is shown. It is parsed exactly like a
search, must keep everything the prompt already asked for, must add what it
claims to add, and must fill at least a full page of results without any
filter being relaxed. So a suggestion never leads to an empty page or to a
"showing other metals instead" note: "for men" is offered only where men's
designs exist (rings, bracelets), "without stones" only where enough plain
designs exist, a stone cut only where enough designs have it.

Prompts with words the search doesn't know get no suggestions: those words may
make it a search for something else entirely ("boxing ring"), which only the
search's domain check (domain.py) can tell.
"""
import re
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .config import CATEGORIES
from .query import _ENTRIES, CONNECTORS, NEG_CUES, SHAPES, STONE_INTENTS, WEAK_CATEGORY_TERMS, parse
from .search import TOP_K

MAX_ITEMS = 8           # suggestions per answer (the page adds the shopper's own history on top)
MAX_COMPLETIONS = 3
MAX_CUTS = 2            # stone cuts offered for one prompt
MAX_TEXT = 200          # longer prompts get no suggestions
COUNT_SHOWN_BELOW = 0.9  # a row shows its number of designs when it keeps at most 90% of the prompt's

# ---- the shopper's script ----------------------------------------------------

_GU, _HI, _LATIN = re.compile(r"[઀-૿]"), re.compile(r"[ऀ-ॿ]"), re.compile(r"[A-Za-z]")


def script(text: str) -> str:
    """"gu" (Gujarati), "hi" (Devanagari) or "en" (English and Hinglish)."""
    counts = [(len(_GU.findall(text)), "gu"), (len(_HI.findall(text)), "hi"), (len(_LATIN.findall(text)), "en")]
    n, which = max(counts)
    return which if n else "en"


# ---- completions: words and phrases the search understands, correctly spelt ----
# In order of preference. Misspellings the parser also accepts ("neckless",
# "pandant") are deliberately missing: suggestions teach the right word.
VOCAB = {
    "en": [
        # type
        "rings", "ring", "earrings", "earring", "stud earrings", "hoop earrings", "drop earrings", "pendants",
        "pendant", "necklaces", "necklace", "bracelets", "bracelet", "bangles", "bangle", "choker", "chain",
        "mangalsutra", "locket", "cuff",
        # metal
        "rose gold", "yellow gold", "white gold",
        # wearer
        "for women", "for men", "for my mother", "for my wife", "for my sister", "for my daughter", "for kids",
        "gents", "ladies",
        # occasion
        "for everyday wear", "everyday", "daily wear", "office wear", "for wedding", "wedding", "bridal",
        "engagement", "for engagement", "party wear", "for party wear", "festive", "diwali", "gift",
        "anniversary gift", "birthday gift",
        # stones
        "with diamonds", "diamonds", "diamond", "without stones", "no stones", "no side stones", "plain gold",
        "solitaire", "single diamond", "halo", "three stone", "side stones", "cluster", "tennis", "pave",
        "bina stone", "heera",
        # stone cut
        "oval", "pear", "princess cut", "emerald cut", "cushion cut", "marquise", "round", "radiant cut",
        "asscher cut", "baguette",
        # band and form
        "thin", "thin band", "wide", "wide band", "split shank", "bezel", "prong", "channel set", "adjustable",
        "open design", "multiple rows", "single row", "drop", "dangling", "chandelier", "stud", "hoop",
        # look and motif
        "floral", "flower", "heart", "infinity", "leaf", "butterfly", "star", "moon", "evil eye", "initial",
        "letter", "bow", "twisted", "geometric", "vintage", "openwork", "filigree", "traditional", "modern",
        "minimal", "simple", "delicate", "dainty", "lightweight", "statement", "bold", "heavy", "chunky",
        "intricate", "medallion", "coin", "charm", "signet", "matte",
        # Hinglish names of pieces
        "anguthi", "haar", "rani haar", "bali", "jhumka", "kada",
    ],
    "gu": [
        "વીંટી", "કાનની બુટ્ટી", "બુટ્ટી", "ઝુમખા", "પેન્ડન્ટ", "લોકેટ", "હાર", "ગળાનો હાર", "રાણી હાર", "ચોકર",
        "મંગળસૂત્ર", "ચેન", "બ્રેસલેટ", "બંગડી", "કડું",
        "રોઝ ગોલ્ડ", "યલો ગોલ્ડ", "વ્હાઇટ ગોલ્ડ", "પીળું સોનું", "સફેદ સોનું",
        "લેડીઝ માટે", "પુરુષો માટે", "મમ્મી માટે", "પત્ની માટે", "બહેન માટે", "દીકરી માટે",
        "રોજ પહેરવા માટે", "લગ્ન માટે", "સગાઈ માટે", "પાર્ટી માટે", "દિવાળી માટે", "ભેટ",
        "હીરા", "હીરા સાથે", "હીરા વગર", "સોલિટેર", "ઓવલ",
        "નાની", "સાદી", "પાતળી", "જાડી", "ભારે", "નકશી", "જાળી", "ફૂલ", "દિલ", "પતંગિયું", "તારો",
        "પરંપરાગત", "મોડર્ન", "લટકતી",
    ],
    "hi": [
        "अंगूठी", "बालियां", "कान की बाली", "झुमके", "पेंडेंट", "लॉकेट", "हार", "रानी हार", "चोकर", "मंगलसूत्र",
        "चेन", "ब्रेसलेट", "कंगन", "कड़ा", "चूड़ियां",
        "रोज़ गोल्ड", "येलो गोल्ड", "व्हाइट गोल्ड", "पीला सोना", "सफेद सोना",
        "लेडीज के लिए", "पुरुषों के लिए", "माँ के लिए", "पत्नी के लिए", "बहन के लिए", "बेटी के लिए",
        "रोज़ पहनने के लिए", "शादी के लिए", "सगाई के लिए", "पार्टी के लिए", "दिवाली के लिए", "गिफ्ट",
        "हीरे", "हीरे के साथ", "हीरे के बिना", "सॉलिटेयर", "ओवल",
        "छोटी", "सादी", "पतली", "मोटी", "भारी", "नक्काशी", "जाली", "फूल", "दिल", "तितली", "पारंपरिक", "मॉडर्न",
        "लटकन",
    ],
}
PLURAL_ONLY = {"rings", "earrings", "pendants", "necklaces", "bracelets", "bangles", "diamonds", "with diamonds"}
ARTICLES = {"a", "an", "one", "ek"}
_KNOWN_WORDS = {w for phrases in VOCAB.values() for p in phrases for w in p.split()}

# ---- refinements -------------------------------------------------------------
# (facet, what it adds, {script: (where, words)}, only for this type or None)
# where: "pre" = in front of the piece's name ("rose gold rings", "a rose gold
# ring for my mother"), "post" = after the prompt ("rings for men").
REFINEMENTS = [
    ("wearer", "intent:women", {"en": ("post", "for women"), "gu": ("post", "લેડીઝ માટે"), "hi": ("post", "लेडीज के लिए")}, None),
    ("wearer", "intent:men", {"en": ("post", "for men"), "gu": ("post", "પુરુષો માટે"), "hi": ("post", "पुरुषों के लिए")}, None),
    ("metal", "metal:rose_gold", {"en": ("pre", "rose gold"), "gu": ("pre", "રોઝ ગોલ્ડ"), "hi": ("pre", "रोज़ गोल्ड")}, None),
    ("metal", "metal:yellow_gold", {"en": ("pre", "yellow gold"), "gu": ("pre", "યલો ગોલ્ડ"), "hi": ("pre", "येलो गोल्ड")}, None),
    ("metal", "metal:white_gold", {"en": ("pre", "white gold"), "gu": ("pre", "વ્હાઇટ ગોલ્ડ"), "hi": ("pre", "व्हाइट गोल्ड")}, None),
    ("stones", "intent:solitaire", {"en": ("pre", "solitaire"), "gu": ("pre", "સોલિટેર"), "hi": ("pre", "सॉलिटेयर")}, None),
    ("stones", "intent:diamond", {"en": ("post", "with diamonds"), "gu": ("post", "હીરા સાથે"), "hi": ("post", "हीरे के साथ")}, None),
    ("stones", "intent:plain", {"en": ("post", "without stones"), "gu": ("post", "હીરા વગર"), "hi": ("post", "हीरे के बिना")}, None),
    ("stones", "intent:halo", {"en": ("pre", "halo")}, "ring"),
    ("form", "intent:thin", {"en": ("pre", "thin"), "gu": ("pre", "પાતળી"), "hi": ("pre", "पतली")}, "ring"),
    ("form", "intent:wide", {"en": ("pre", "wide")}, "ring"),
    ("form", "intent:stud", {"en": ("pre", "stud")}, "earrings"),
    ("form", "intent:drop", {"en": ("pre", "drop"), "gu": ("pre", "લટકતી")}, "earrings"),
    ("form", "intent:hoop", {"en": ("pre", "hoop")}, "earrings"),
    ("form", "intent:cuff", {"en": ("pre", "cuff")}, "bracelet"),
    ("form", "intent:bangle", {"en": ("pre", "bangle")}, "bracelet"),
    ("form", "intent:choker", {"en": ("pre", "choker")}, "necklace"),
    ("occasion", "intent:everyday", {"en": ("post", "for everyday wear"), "gu": ("post", "રોજ પહેરવા માટે"), "hi": ("post", "रोज़ पहनने के लिए")}, None),
    ("occasion", "intent:bridal", {"en": ("post", "for wedding"), "gu": ("post", "લગ્ન માટે"), "hi": ("post", "शादी के लिए")}, None),
    ("occasion", "intent:engagement", {"en": ("post", "for engagement"), "gu": ("post", "સગાઈ માટે"), "hi": ("post", "सगाई के लिए")}, "ring"),
    ("occasion", "intent:party", {"en": ("post", "for party wear"), "gu": ("post", "પાર્ટી માટે"), "hi": ("post", "पार्टी के लिए")}, None),
]
# a facet is skipped when the prompt already says something about it
FACET_INTENTS = {
    "wearer": {"men", "women", "kids"},
    "stones": STONE_INTENTS | {"plain"},
    "form": {"thin", "wide", "stud", "hoop", "drop", "jhumka", "chandelier", "bangle", "cuff", "flexible", "choker",
             "pendant_chain", "rani_haar"},
    "occasion": {"everyday", "bridal", "party", "engagement", "gift"},
}
FACET_ORDER = ["wearer", "metal", "stones", "form", "occasion", "cut"]
TYPE_WORDS = {
    "en": {"ring": ("ring", "rings"), "earrings": ("earring", "earrings"), "pendant": ("pendant", "pendants"),
           "necklace": ("necklace", "necklaces"), "bracelet": ("bracelet", "bracelets")},
    "gu": {"ring": "વીંટી", "earrings": "બુટ્ટી", "pendant": "પેન્ડન્ટ", "necklace": "હાર", "bracelet": "બ્રેસલેટ"},
    "hi": {"ring": "अंगूठी", "earrings": "बालियां", "pendant": "पेंडेंट", "necklace": "हार", "bracelet": "ब्रेसलेट"},
}
CUT_EN = {"emerald": "emerald cut", "princess": "princess cut", "cushion": "cushion cut", "radiant": "radiant cut",
          "asscher": "asscher cut"}
# cut words in Gujarati / Devanagari script, from the parser's own list
CUT_WORDS = {
    "en": {s: CUT_EN.get(s, s) for s in SHAPES},
    "gu": {s: next((t for t in terms if _GU.search(t)), None) for s, terms in SHAPES.items()},
    "hi": {s: next((t for t in terms if _HI.search(t)), None) for s, terms in SHAPES.items()},
}
# English words that start a phrase after the piece's name ("rings for my mother")
LEAD_PREPOSITIONS = {"for", "with", "without", "in", "under", "as", "to"}
# a prompt ending in one of these is mid-phrase: complete it, don't append to it
DANGLING = {"for", "with", "without", "in", "and", "or", "no", "not", "of", "a", "an", "the", "my", "to", "under",
            "bina", "ke", "ki", "ka"}
# English filler in front of a piece's name: new describing words go after it
# ("a | rose gold ring", "show me | rose gold bold rings")
FILLER = {"a", "an", "the", "i", "me", "my", "we", "us", "our", "you", "your", "want", "need", "show", "find",
          "get", "give", "looking", "look", "for", "like", "would", "some", "any", "please", "with", "of", "in",
          "on", "to", "at", "is", "are", "which", "that", "this", "these", "her", "his", "their", "him", "she",
          "he", "who", "something", "piece", "chahiye", "mujhe", "koi", "ek", "ke", "ki", "ka", "liye", "mein",
          "wala", "wali"}
# ...except after a joining or negating word: "simple and thin | rose gold ring",
# never "no | rose gold stone ring" (that would negate the metal)
JOINERS = CONNECTORS | {w for c in NEG_CUES for w in c.split() if w.isascii()}
_GOLD = re.compile(r"gold|sona|સોન|ગોલ્ડ|सोन|गोल्ड")
_PUNCT = re.compile(r"[+&(),.;:!?\"/\\\-_|*]")


def _sig(p):
    return (frozenset(p.categories), p.metal, frozenset(p.exclude_categories), frozenset(p.exclude_metals),
            p.shape, frozenset(p.intents), frozenset(p.not_intents))


def _keeps(base, cand) -> bool:
    """Everything the prompt asked for (and against) is still there."""
    return (set(base.categories) <= set(cand.categories) and base.metal in (None, cand.metal)
            and set(base.exclude_categories) <= set(cand.exclude_categories)
            and set(base.exclude_metals) <= set(cand.exclude_metals) and base.shape in (None, cand.shape)
            and set(base.intents) <= set(cand.intents) and set(base.not_intents) <= set(cand.not_intents))


def _has(p, key: str) -> bool:
    kind, value = key.split(":", 1)
    return {"cat": lambda: value in p.categories, "metal": lambda: p.metal == value,
            "intent": lambda: value in p.intents, "shape": lambda: p.shape == value}[kind]()


def _understood(p) -> bool:
    return bool(p.categories or p.metal or p.intents or p.not_intents or p.shape or p.exclude_categories
                or p.exclude_metals)


def _noun_span(text: str):
    """Where the piece's name ("ring", "કાનની બુટ્ટી") starts and ends in the
    text, found the way the parser finds it (longest terms first, each
    consuming its span), or None."""
    t = _PUNCT.sub(" ", text.lower().replace("’", "'"))
    if len(t) != len(text):
        return None
    t = " " + t + " "
    strong, weak = [], []
    for _, kind, _, _, term, rx in _ENTRIES:
        m = rx.search(t)
        while m:
            if kind == "category":
                (weak if term in WEAK_CATEGORY_TERMS else strong).append((m.start() - 1, m.end() - 1))
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
            m = rx.search(t)
    found = strong or weak
    return min(found) if found else None


def _fix_article(before: str, word: str) -> str:
    """"an" + "rose gold earring" -> "a rose gold earring"."""
    m = re.search(r"(?:^|\s)(an?)\s*$", before, re.I)
    if not m:
        return before
    right = "an" if word[:1].lower() in "aeiou" else "a"
    if m.group(1).isupper():
        right = right.upper()
    elif m.group(1)[0].isupper():
        right = right.capitalize()
    return before[:m.start(1)] + right + before[m.end(1):]


def _insert_pre(text: str, words: str, lang: str) -> str:
    """Put words in front of the piece's name. English: before the describing
    words, after any filler ("rose gold bold rings", "a rose gold ring for my
    mother"). Gujarati / Hindi: right before the name ("... સાદી રોઝ ગોલ્ડ
    કાનની બુટ્ટી"). With no name in the prompt: at the start."""
    span = _noun_span(text)
    if span is None:
        m = re.match(r"\s*(?:an?|the)\s+", text, re.I) if lang == "en" else None
        at = m.end() if m else 0
    elif lang != "en":
        at = span[0]
    else:
        at = 0
        for w in re.finditer(r"\S+", text[:span[0]]):
            word = _PUNCT.sub("", w.group().lower())
            if word in JOINERS:
                at = span[0]
            elif word in FILLER:
                at = w.end()
    before, after = text[:at], text[at:]
    if lang == "en":
        before = _fix_article(before, words)
    return f"{before.rstrip()} {words} {after.lstrip()}".strip()


def _append(text: str, words: str) -> str:
    return re.sub(r"[\s,.;:!?\-]+$", "", text) + " " + words


def _place(text: str, where: str, words: str, lang: str) -> str:
    return _insert_pre(text, words, lang) if where == "pre" else _append(text, words)


def _add_type(text: str, cat: str, lang: str) -> str:
    """A piece's name added to a prompt that has none: "floral" -> "floral
    rings", "for my mother" -> "rings for my mother", "ફૂલ" -> "ફૂલ વીંટી"."""
    if lang != "en":
        return _append(text, TYPE_WORDS[lang][cat])
    words = text.split()
    one, many = TYPE_WORDS["en"][cat]
    noun = one if words and words[0].lower() in ARTICLES else many
    if words and words[0].lower() in LEAD_PREPOSITIONS:
        return f"{noun} {text.strip()}"
    return _append(text, noun)


def _corrected(text: str, keep_last: bool) -> str:
    """The prompt with the parser's spelling corrections applied ("bracelate
    for party" -> "bracelet for party"); a word still being typed is left alone."""
    head, tail = text, ""
    if keep_last:
        m = re.search(r"\S+$", text)
        if m:
            head, tail = text[:m.start()], m.group()
    for wrong, right in parse(head).corrections.items():
        head = re.sub(r"(?<![A-Za-z0-9])" + re.escape(wrong) + r"(?![A-Za-z0-9])", right, head, flags=re.I)
    return head + tail


class Ctx(NamedTuple):
    """Words typed next to a photo refine the photo: no counts are shown (the
    photo decides the order), and the photo's type, when its reading is sure,
    counts as the type unless the words name one (as in search_photo)."""
    photo: bool = False
    ptype: str | None = None

    def parse(self, text: str):
        q = parse(text)
        if self.ptype and not q.categories:
            q.categories, q.category = [self.ptype], self.ptype
        return q


class Suggester:
    def __init__(self, engine):
        self.engine = engine
        self._cached = lru_cache(maxsize=4096)(self._suggest)

    def __call__(self, text: str, photo: bool = False, category: str | None = None) -> dict:
        """photo: the words refine a photo; category: that photo's type, when sure."""
        text = re.sub(r"\s+", " ", text or "").lstrip()
        if not text.strip() or len(text) > MAX_TEXT:
            return {"items": []}
        return self._cached(text, Ctx(bool(photo), category if photo and category in CATEGORIES else None))

    # ---- checks ----
    def _check(self, base, base_n: int, text: str, claim, ctx: Ctx, same_ok: bool = False):
        """-> a suggestion dict, or None when the text would not be a good search.
        same_ok: it may mean what the base means (a half-typed word finished)."""
        cand = ctx.parse(text)
        if (_sig(cand) == _sig(base) and not same_ok) or not _keeps(base, cand):
            return None, cand
        if cand.free_text():
            return None, cand    # words the search doesn't know could make it a non-jewellery search (domain.py)
        if claim and not _has(cand, claim):
            return None, cand
        if claim and not claim.startswith("cat:") and set(cand.categories) != set(base.categories):
            return None, cand    # the refinement words named another type by accident
        relaxed = []
        mask, _ = self.engine._filters(cand, relaxed)
        n = int(mask.sum())
        if relaxed or n < TOP_K:
            return None, cand
        # a stone cut only reorders results, so enough designs must really have it
        if cand.shape and int((mask & (self.engine.shape == cand.shape)).sum()) < TOP_K:
            return None, cand
        # the count is the designs the filters keep; with a cut that would
        # overstate it (the cut only reorders them), so no number is shown then
        show = not ctx.photo and not cand.shape and n <= COUNT_SHOWN_BELOW * base_n
        return {"text": text, "count": n if show else None}, cand

    # ---- the three sources ----
    def _completions(self, text: str, ctx: Ctx):
        """The word being typed, finished: "bold ri" -> "bold rings",
        "rose g" -> "rose gold", "rings for " -> "rings for men"."""
        ends_space = text.endswith(" ")
        stripped = text.rstrip()
        starts = [m.start() for m in re.finditer(r"\S+", stripped)]
        if not starts:
            return []
        typed = _sig(ctx.parse(stripped))
        # a half-typed last word is finished even when that adds no meaning: "hoop" already
        # means hoop earrings, yet "hoop ear" should become "hoop earrings" ("ring" -> "rings" not)
        last = stripped[starts[-1]:].lower()
        unfinished = not ends_space and last not in _KNOWN_WORDS
        out, seen = [], set() if unfinished else {typed}
        for k in (3, 2, 1):   # a longer typed tail first: "rose g" -> "rose gold"
            if k > len(starts):
                continue
            head = stripped[:starts[-k]]
            tail = (stripped[starts[-k]:] + (" " if ends_space else "")).lower()
            after_article = k == 1 and len(starts) >= 2 and stripped[starts[-2]:starts[-1]].strip().lower() in ARTICLES
            head_p = ctx.parse(head)
            head_n, _ = self.engine.count(head_p)
            for phrase in VOCAB[script(tail)]:
                if not phrase.startswith(tail) or phrase == tail.strip():
                    continue
                if after_article and phrase in PLURAL_ONLY:   # "a ri" -> "a ring", not "a rings"
                    continue
                same_ok = unfinished and _understood(ctx.parse(phrase))   # never a bare "with"
                item, cand = self._check(head_p, head_n, head + phrase, None, ctx, same_ok)
                if item and _sig(cand) not in seen:
                    seen.add(_sig(cand))
                    out.append({**item, "kind": "complete"})
                    if len(out) >= MAX_COMPLETIONS:
                        return out
            if out:
                return out
        return out

    def _refinements(self, text: str, ctx: Ctx, limit: int):
        last = re.sub(r"[^\w]", "", text.split()[-1].lower()) if text.split() else ""
        if not _understood(parse(text)) or parse(text).free_text() or last in DANGLING:
            return []   # nothing known, unknown words (maybe not jewellery at all), or mid-phrase
        base = ctx.parse(text)
        lang = script(text)
        mask, _ = self.engine._filters(base)
        base_n = int(mask.sum())
        facets: dict[str, list] = {}
        one_type = base.categories[0] if len(base.categories) == 1 else None
        # no piece named (words next to a photo, "bina stone"): describing words
        # go after the prompt ("rose gold stud", not "stud rose gold"); a metal still leads
        nameless = _noun_span(text) is None

        def add(facet, cand_text, claim):
            item, _ = self._check(base, base_n, cand_text, claim, ctx)
            if item:
                item["kind"] = "refine"
                facets.setdefault(facet, []).append(item)

        # the type comes first when the prompt names none (a photo gives its own)
        if not base.categories and not base.exclude_categories and not ctx.photo:
            for c in sorted(CATEGORIES, key=lambda c: -int((self.engine.cat == c).sum())):
                add("type", _add_type(text, c, lang), "cat:" + c)
        for facet, claim, words, only in REFINEMENTS:
            if lang not in words or (only and only != one_type):
                continue
            if facet == "metal" and (base.metal or base.exclude_metals or _GOLD.search(text.lower())):
                continue   # "plain gold" would become "rose gold plain gold"
            if facet in FACET_INTENTS and FACET_INTENTS[facet] & (set(base.intents) | set(base.not_intents)):
                continue
            where, w = words[lang]
            if nameless and facet != "metal":
                where = "post"
            add(facet, _place(text, where, w, lang), claim)
        # stone cuts that enough designs here really have (from the design ids)
        if one_type and not base.shape and "plain" not in base.intents:
            cuts = [(int((mask & (self.engine.shape == s)).sum()), s) for s in SHAPES]
            for n, s in sorted(cuts, reverse=True)[:MAX_CUTS]:
                w = CUT_WORDS[lang].get(s)
                if n >= TOP_K and w:
                    add("cut", _place(text, "post" if nameless else "pre", w, lang), "shape:" + s)

        # a prompt naming no type gets the types only ("floral" -> "floral
        # rings", "floral earrings", ...); otherwise one value of each facet per round
        if facets.get("type"):
            return facets["type"][:limit]
        out = []
        queues = [facets[f] for f in FACET_ORDER if f in facets]
        while len(out) < limit and any(queues):
            for q in queues:
                if q and len(out) < limit:
                    out.append(q.pop(0))
        return out[:limit]

    @staticmethod
    def _still_typing(text: str) -> bool:
        """The last word adds nothing yet and starts a word the search knows
        without being one: "gents ear" (earrings), "rings go" (gold)."""
        words = text.split()
        if not words or _sig(parse(text)) != _sig(parse(" ".join(words[:-1]))):
            return False
        last = words[-1].lower()
        return last not in _KNOWN_WORDS and any(w.startswith(last) for w in _KNOWN_WORDS)

    def _suggest(self, text: str, ctx: Ctx) -> dict:
        mid_word = not text.endswith(" ")
        text_c = _corrected(text, keep_last=mid_word)
        items = self._completions(text_c, ctx)
        if mid_word and not items and self._still_typing(text_c):
            return {"items": []}   # "gents ear": still typing, and no finished word fits
        if text_c.strip().lower() != text.strip().lower() and not items:
            # the prompt with its spelling fixed, as the first suggestion
            items = [{"text": text_c.strip(), "count": None, "kind": "fix"}]
        # refine the best completion when there is one ("bold ri" -> "bold rings for men")
        base_text = items[0]["text"] if items else text_c.strip()
        items += self._refinements(base_text, ctx, MAX_ITEMS - len(items))
        seen, out = {text.strip().lower()}, []
        for it in items:
            key = re.sub(r"\s+", " ", it["text"].strip().lower())
            if key not in seen:
                seen.add(key)
                out.append({**it, "text": it["text"].strip()})
        return {"items": out[:MAX_ITEMS]}
