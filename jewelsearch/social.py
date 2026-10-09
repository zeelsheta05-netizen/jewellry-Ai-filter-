"""Social post captions for a jewellery picture (AI generation panel, "Social post" tab).

One uploaded picture -> one caption per chosen platform, written for that platform's
rules (length, hook, hashtag count, title fields) and made search friendly:

1. Facts: our own photo reading (Design DNA: piece type, metal, stones, style traits),
   plus the details the user typed (carat, hallmark, price...). Captions may only claim
   what is in these facts: sentences with unlisted claims (discounts, certificates,
   carats, prices...) are dropped after writing.
2. Trends: a dated festival and season calendar (Diwali, Dhanteras, wedding season,
   Valentine's Day...) gives the timely angle, keywords and hashtags for today's date.
   This is a seasonal calendar, not live social-network trend data.
3. SEO: a main keyword built from the facts ("rose gold diamond ring") goes in the
   first line / title; related search phrases and hashtags are built here, not by the
   writer, so they are always on-topic and within each platform's limit.
4. Writer: Pollinations' text model (looks at the picture) when connected, else the
   local Qwen3-1.7B the app already has loaded, else a ready-made template.
"""
from __future__ import annotations

import datetime as dt
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from . import sketch
from .sketch import SketchError


# ---------- platforms ----------

@dataclass(frozen=True)
class Platform:
    key: str
    label: str
    limit: int            # characters for the whole post (text + hashtags)
    tags: int             # hashtags to add
    tokens: int           # writer answer length
    guide: str            # how a post on this platform should read
    title: int = 0        # >0: the platform has a title field of this many characters
    hook: int = 0         # >0: the first line must fit in this many characters (shown before "more")


PLATFORMS: dict[str, Platform] = {p.key: p for p in [
    Platform("instagram", "Instagram", 2200, 5, 220,
             "Start with a short, striking first line that names the piece with the main keyword. Then 2 to 4 short lines "
             "about the design, how it feels to wear it and the occasion. Short paragraphs, friendly.", hook=125),
    Platform("facebook", "Facebook", 1200, 3, 200,
             "Warm and conversational, 2 to 4 short sentences. Mention the main keyword early. "
             "End with a question that invites comments.", hook=120),
    Platform("x", "X (Twitter)", 270, 2, 90,   # 280, less room for emojis counted twice
             "One or two punchy sentences, under 200 characters in total. Main keyword included."),
    Platform("threads", "Threads", 500, 1, 140,
             "Casual, like talking to a friend, 2 or 3 short sentences. Main keyword included."),
    Platform("linkedin", "LinkedIn", 3000, 3, 260,
             "Professional: the craftsmanship, the design idea and the team behind it, in 2 or 3 short "
             "paragraphs. No slang. Main keyword in the first sentence.", hook=140),
    Platform("pinterest", "Pinterest", 500, 2, 170,
             "A keyword-rich description of 2 or 3 sentences that says what the piece is, its style and "
             "who or what occasion it suits, using natural search phrases.", title=100),
    Platform("youtube", "YouTube Shorts", 1000, 3, 170,
             "A short description of 2 or 3 sentences for a jewellery Short, main keyword in the first sentence.",
             title=100),
    Platform("google", "Google Business", 1500, 0, 180,
             "A local business update for the shop's Google profile: 2 or 3 sentences, the main keyword "
             "and the city (if given) for local search. No hashtags, no emojis."),
]}

TONES = {
    "elegant": ("Elegant", "elegant and refined"),
    "festive": ("Festive", "joyful and festive"),
    "romantic": ("Romantic", "romantic and heartfelt"),
    "playful": ("Playful", "playful and fun"),
    "minimal": ("Minimal", "calm, simple and modern"),
    "luxury": ("Bold luxury", "bold, confident and luxurious"),
    "professional": ("Professional", "professional and informative"),
}
GOALS = {
    "sell": ("Sell / promote", "make people want to buy or enquire"),
    "launch": ("New launch", "announce a new design"),
    "engage": ("Get comments", "start a conversation and get comments"),
    "story": ("Brand story", "tell the craft story behind the piece"),
}
CTAS = {
    "dm": "DM us to order",
    "link": "Shop now, link in bio",
    "store": "Visit our store to see it in person",
    "whatsapp": "WhatsApp us to know more",
    "custom": "Ask us to make it your way",
    "none": "",
}
MARKETS = {"in": "India", "global": "Global"}

DETAILS_MAX, BRAND_MAX, CITY_MAX, CTA_MAX = 400, 60, 40, 120


# ---------- trends: festival and season calendar ----------

@dataclass(frozen=True)
class Occasion:
    key: str
    label: str
    day: dt.date          # the day itself (or a season's last day)
    lead: int             # shown this many days before
    markets: tuple
    keywords: tuple       # search phrases, {kind} = ring / earrings / ...
    tags: tuple           # hashtags without '#'
    angle: str            # one line for the writer
    start: dt.date | None = None   # a season: from this day
    weight: int = 5                # how much it moves jewellery sales (picks the "auto" angle)
    yearly: bool = False           # same calendar date every year


def _d(y, m, d):
    return dt.date(y, m, d)


IN, ALL = ("in",), ("in", "global")
# 2026-27 dates checked 2026-10-09 (lunar festivals can move a day by region).
OCCASIONS: list[Occasion] = [
    Occasion("festive26", "Festive season", _d(2026, 11, 15), 0, IN,
             ("festive jewellery", "festive {kind}", "jewellery for festive season"),
             ("FestiveJewellery", "FestiveSeason", "FestiveLook"), "the festive season of Navratri, Dussehra and Diwali",
             start=_d(2026, 10, 1), weight=6),
    Occasion("navratri", "Navratri", _d(2026, 10, 19), 14, IN, ("navratri jewellery", "festive {kind}"),
             ("Navratri", "NavratriLook", "FestiveJewellery"), "the nine nights of Navratri and festive looks",
             start=_d(2026, 10, 11), weight=5),
    Occasion("dussehra", "Dussehra", _d(2026, 10, 20), 14, IN, ("dussehra jewellery", "festive {kind}"),
             ("Dussehra", "FestiveJewellery"), "Dussehra celebrations and new beginnings", weight=4),
    Occasion("karwachauth", "Karwa Chauth", _d(2026, 10, 29), 21, IN,
             ("karwa chauth gift", "karwa chauth jewellery", "gift for wife"),
             ("KarwaChauth", "KarwaChauthGift", "GiftForWife"), "Karwa Chauth and a gift for her", weight=7),
    Occasion("dhanteras", "Dhanteras", _d(2026, 11, 6), 30, IN,
             ("dhanteras gold buying", "dhanteras jewellery", "gold {kind} for dhanteras"),
             ("Dhanteras", "Dhanteras2026", "GoldJewellery"), "Dhanteras, the most auspicious day to buy gold", weight=10),
    Occasion("diwali", "Diwali", _d(2026, 11, 8), 35, IN,
             ("diwali jewellery", "diwali gift for her", "festive {kind}"),
             ("Diwali", "Diwali2026", "DiwaliGifts", "FestiveJewellery"), "Diwali sparkle, gifting and festive outfits", weight=10),
    Occasion("bhaidooj", "Bhai Dooj", _d(2026, 11, 11), 8, IN, ("bhai dooj gift for sister",),
             ("BhaiDooj", "GiftForSister"), "a Bhai Dooj gift for a sister", weight=3),
    Occasion("wedding26", "Wedding season", _d(2027, 2, 28), 10, IN,
             ("bridal jewellery", "wedding jewellery", "bridal {kind}"),
             ("BridalJewellery", "WeddingSeason", "IndianBride"), "the wedding season and bridal looks",
             start=_d(2026, 11, 12), weight=8),
    Occasion("blackfriday", "Black Friday", _d(2026, 11, 27), 10, ("global",), ("jewellery gift", "{kind} gift"),
             ("BlackFriday", "GiftIdeas"), "Black Friday gifting", weight=6),
    Occasion("christmas", "Christmas", _d(2026, 12, 25), 25, ALL, ("christmas gift for her", "christmas jewellery gift"),
             ("ChristmasGifts", "GiftForHer"), "Christmas gifting", weight=7, yearly=True),
    Occasion("newyear", "New Year", _d(2027, 1, 1), 8, ALL, ("new year jewellery", "party {kind}"),
             ("NewYearLook", "PartyJewellery"), "New Year's Eve party looks", weight=5, yearly=True),
    Occasion("valentine", "Valentine's Day", _d(2027, 2, 14), 21, ALL,
             ("valentine's day gift for her", "{kind} gift for girlfriend"),
             ("ValentinesDay", "ValentinesGift", "GiftForHer"), "Valentine's Day and saying it with jewellery", weight=8, yearly=True),
    Occasion("womensday", "Women's Day", _d(2027, 3, 8), 10, ALL, ("women's day gift", "self-gift jewellery"),
             ("WomensDay", "SelfLove"), "Women's Day and treating yourself", weight=4, yearly=True),
    Occasion("eid", "Eid", _d(2027, 3, 10), 14, ALL, ("eid jewellery", "eid gift"),
             ("Eid", "EidMubarak", "EidLook"), "Eid celebrations and festive looks", weight=6),
    Occasion("akshaya", "Akshaya Tritiya", _d(2027, 5, 8), 21, IN,
             ("akshaya tritiya gold", "akshaya tritiya jewellery"), ("AkshayaTritiya", "GoldJewellery"),
             "Akshaya Tritiya, an auspicious day to buy gold", weight=9),
    Occasion("mothersday", "Mother's Day", _d(2027, 5, 9), 18, ALL, ("mother's day gift", "jewellery gift for mom"),
             ("MothersDay", "GiftForMom"), "Mother's Day gifting", weight=6),
    Occasion("wedding27", "Summer wedding season", _d(2027, 6, 15), 0, IN, ("bridal jewellery", "wedding {kind}"),
             ("BridalJewellery", "WeddingSeason"), "the summer wedding season", start=_d(2027, 4, 15), weight=8),
    Occasion("rakhi", "Raksha Bandhan", _d(2027, 8, 17), 18, IN, ("raksha bandhan gift for sister",),
             ("RakshaBandhan", "RakhiGift", "GiftForSister"), "a Raksha Bandhan gift for a sister", weight=5),
]
OCCASION_BY_KEY = {o.key: o for o in OCCASIONS}


def trending(today: dt.date | None = None, market: str = "in") -> list[Occasion]:
    """Occasions people are searching and posting about around this date, nearest first."""
    today = today or dt.date.today()
    out = []
    for o in OCCASIONS:
        if market not in o.markets:
            continue
        for shift in range(0, (today.year - o.day.year + 2) if o.yearly else 1):
            day = o.day.replace(year=o.day.year + shift)
            first = (o.start.replace(year=o.start.year + shift) if o.start else day) - dt.timedelta(days=o.lead)
            if first <= today <= day:
                out.append(o)
                break
    # the occasions that sell the most jewellery first, then the nearest
    return sorted(out, key=lambda o: (-o.weight, o.day))


# ---------- facts ----------

KINDS = {"earrings": "earrings", "ring": "ring", "pendant": "pendant", "necklace": "necklace",
         "bracelet": "bracelet", "bangle": "bangle", "mangalsutra": "mangalsutra", "nosepin": "nose pin",
         "chain": "chain", "anklet": "anklet"}
KIND_WORDS = [   # the user's own words for the piece (checked in this order)
    (r"\bmangal ?sutras?\b", "mangalsutra"), (r"\bnose ?pins?\b|\bnath\b", "nose pin"),
    (r"\bear ?rings?\b|\bstuds?\b|\bjhumkas?\b|\bhoops?\b|\bdanglers?\b", "earrings"),
    (r"\bpendants?\b", "pendant"), (r"\bnecklaces?\b|\bchokers?\b", "necklace"),
    (r"\bbangles?\b|\bkadas?\b", "bangle"), (r"\bbracelets?\b", "bracelet"), (r"\banklets?\b|\bpayals?\b", "anklet"),
    (r"\brings?\b", "ring"), (r"\bchains?\b", "chain"),
]
STONE_WORDS = ("diamond", "emerald", "ruby", "sapphire", "pearl", "polki", "kundan", "moissanite", "tanzanite",
               "topaz", "amethyst", "opal", "garnet", "zircon", "cz")
METAL_WORDS = ("rose gold", "white gold", "yellow gold", "platinum", "sterling silver", "silver", "gold")


@dataclass
class Facts:
    kind: str = "jewellery"          # ring, earrings, ...
    metal: str = ""                  # rose gold ...
    stone: str = ""                  # diamond / emerald / "" ...
    stone_style: str = ""            # solitaire, side stones, many diamonds, no stones
    traits: list[str] = field(default_factory=list)
    details: str = ""                # what the user typed (trusted, may be quoted)
    sure: bool = True

    def keyword(self) -> str:
        """The main search phrase: "rose gold diamond ring"."""
        stone = self.stone or ("solitaire" if self.stone_style == "solitaire" else "")
        return " ".join(w for w in (self.metal, stone, self.kind) if w)

    def text(self) -> str:
        bits = [self.keyword()]
        if self.stone_style and self.stone_style not in self.keyword():
            bits.append(self.stone_style)
        bits += self.traits[:5]
        return ", ".join(b for b in bits if b)


def read_facts(engine, picture: bytes | None, details: str = "") -> Facts:
    """Design DNA of the picture (local models) + the user's own words, which win."""
    f = Facts(details=details)
    if engine is not None and picture:
        try:
            from . import photo
            d = engine.read_photo(photo.read(picture)).dna or {}
        except Exception as e:
            print(f"social: photo reading failed ({e!r})", flush=True)
            d = {}
        t = d.get("type") or {}
        if t.get("value"):
            f.kind = KINDS.get(t["value"], str(t["value"]).replace("_", " "))
            f.sure = bool(t.get("sure", True))
        f.metal = ((d.get("metal") or {}).get("metal") or "").replace("_", " ")
        traits = d.get("traits") or []
        stones = [x.get("label", "") for x in traits if x.get("group") == "Stones"]
        if stones:
            f.stone_style = stones[0].lower()
            if "diamond" in f.stone_style:
                f.stone = "diamond"
        f.traits = [x.get("label", "").lower() for x in traits if x.get("group") != "Stones" and x.get("label")][:4]
        f.traits += [m.get("label", "").lower() for m in (d.get("motifs") or []) if m.get("label")][:3]
    words = details.lower()
    for rx, kind in KIND_WORDS:
        if re.search(rx, words):
            f.kind, f.sure = kind, True
            break
    for m in METAL_WORDS:
        if m in words:
            f.metal = m
            break
    for s in STONE_WORDS:
        if re.search(rf"\b{s}s?\b", words):
            f.stone = "CZ" if s in ("cz", "zircon") else s
            break
    if f.stone_style == "no stones" and f.stone:
        f.stone_style = ""
    return f


# ---------- SEO keywords and hashtags ----------

def _tag(phrase: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", phrase.replace("'", ""))
    return "#" + "".join(w[:1].upper() + w[1:] for w in words) if words else ""


def spell(text: str, market: str) -> str:
    return text.replace("jewellery", "jewelry").replace("Jewellery", "Jewelry") if market == "global" else text


def keywords(f: Facts, occasions: list[Occasion], city: str = "", market: str = "in") -> list[str]:
    """Search phrases for this post, most specific first (no repeats)."""
    k = f.kind
    out = [f.keyword()]
    if f.metal and f.stone:
        out.append(f"{f.stone} {k}")
    if f.metal:
        out.append(f"{f.metal} {k}")
    if f.stone_style == "solitaire" and k == "ring":
        out.append("solitaire engagement ring")
    for o in occasions[:2]:
        out += [x.format(kind=k) for x in o.keywords[:2]]
    if city:
        out.append(f"jewellery shop in {city}")
    out += [f"{k} design for women", f"{k} gift for her", "fine jewellery"]
    seen, res = set(), []
    for x in out:
        x = spell(" ".join(x.split()).lower(), market)
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res[:8]


def hashtags(f: Facts, occasions: list[Occasion], brand: str = "", city: str = "", market: str = "in") -> list[str]:
    """Hashtag pool, best first: brand, the piece itself, the occasion, then the niche."""
    k = f.kind.replace(" ", "")
    jw = "Jewelry" if market == "global" else "Jewellery"
    pool = [_tag(brand)] if brand else []
    pool += [_tag(f.keyword())]
    if f.stone:
        pool.append(_tag(f"{f.stone} {k}"))
    for o in occasions[:2]:
        pool += ["#" + t for t in o.tags[:2]]
    if f.metal:
        pool.append(_tag(f"{f.metal} {jw}"))
    if city:
        pool.append(_tag(f"{city} {jw}"))
    pool += [_tag(f"{k}s") if not k.endswith("s") else _tag(k), f"#Fine{jw}", f"#{jw}Design", f"#{jw}OfTheDay",
             f"#{jw}Lover"]
    seen, res = set(), []
    for t in pool:
        if len(t) > 2 and t.lower() not in seen and len(t) <= 40:
            seen.add(t.lower())
            res.append(t)
    return res


def pick_tags(pool: list[str], n: int) -> list[str]:
    return pool[:n]


# ---------- honesty: drop claims the facts don't back ----------

CLAIMS = [   # kinds of claim a writer must not make up; allowed when the jeweller's details make it too
    r"\d+\s*%|\bdiscount|\bon sale\b|\bsale\b|\bcoupon|\bpromo code|\bspecial offer|\bflat \d+",
    r"\bcertif", r"\bgia\b", r"\bigi\b", r"\bhallmark|\bbis\b", r"\bsgl\b|\bhrd\b",
    r"\bnatural (diamond|stone|gem)|\blab[- ]?(grown|created)|\bconflict[- ]free|\bethical",
    r"\bfree (shipping|delivery)|\bdelivery\b|\bships?\b",
    r"[₹$€£]|\brs\.?\s*\d|\binr\b|\bprice[ds]?\b",
    r"\b\d+(\.\d+)?\s*(ct|cts|carats?|kt|k|karat|grams?|gm|g|mm)\b",
    r"\blimited (stock|edition)|\bonly \d+ left|\blast few|\bsold out",
    r"\bhand[- ]?made\b|\bhand[- ]?crafted\b|\bhandcrafted",
    r"\bwarranty|\bguarantee|\blifetime\b|\bbuy[- ]?back|\bexchange\b|\bemi\b",
    r"\bartisans?\b|\bcraftsm[ae]n|\bkarigars?\b|\byears of\b|\bsince (19|20)\d\d|\bgenerations?\b|\bbespoke\b|\bcustomi[sz]",
    r"\binvest(ment|ing)?\b|\b(good|great|high|best) returns\b|\bvalue grows",
]


def _allowed(rx: str, sentence: str, details: str) -> bool:
    """The details make the same kind of claim, with the same numbers."""
    if not re.search(rx, details):
        return False
    nums = set(re.findall(r"\d+(?:\.\d+)?", " ".join(m.group(0) for m in re.finditer(rx, sentence))))
    return all(n in re.findall(r"\d+(?:\.\d+)?", details) for n in nums)


def scrub(text: str, details: str) -> tuple[str, list[str]]:
    """Drop sentences that claim something the facts don't say (prices, certificates,
    discounts, carats...). Returns (text, the dropped sentences)."""
    lines, dropped, details = [], [], details.lower()
    for line in text.split("\n"):
        keep = []
        for s in re.split(r"(?<=[.!?])\s+", line):
            low = s.lower()
            bad = any(re.search(rx, low) and not _allowed(rx, low, details) for rx in CLAIMS)
            (dropped if bad else keep).append(s)
        lines.append(" ".join(keep))
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return out, dropped


# ---------- writing ----------

@dataclass
class Brief:
    facts: Facts
    platforms: list[str]
    tone: str = "elegant"
    goal: str = "sell"
    cta: str = "dm"
    cta_text: str = ""
    brand: str = ""
    city: str = ""
    market: str = "in"
    occasion: str = "auto"        # auto / none / an occasion key
    emojis: bool = True
    today: dt.date | None = None

    def occasions(self) -> list[Occasion]:
        if self.occasion == "none":
            return []
        if self.occasion in OCCASION_BY_KEY:
            return [OCCASION_BY_KEY[self.occasion]]
        return trending(self.today, self.market)

    def call(self) -> str:
        if self.cta_text.strip():
            return " ".join(self.cta_text.split())[:CTA_MAX]
        return CTAS.get(self.cta, "")


def _clean_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^\s*(caption|post|here('s| is)[^:\n]*)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"(?m)^\s*#\w+(\s+#\w+)*\s*$", "", text)       # hashtag-only lines: we add our own
    text = re.sub(r"[ \t]*(?<!\w)#\w+", "", text)                  # and any hashtag inside a line
    return re.sub(r"\n{3,}", "\n\n", text).strip().strip('"').strip()


def prompt_for(b: Brief, p: Platform, keys: list[str], extra: str = "") -> str:
    f, occ = b.facts, b.occasions()
    tone = TONES.get(b.tone, TONES["elegant"])[1]
    goal = GOALS.get(b.goal, GOALS["sell"])[1]
    lines = [
        f"Write a {p.label} post for a jewellery brand{' called ' + b.brand if b.brand else ''}.",
        f"The piece (from the photo): {f.text()}.",
    ]
    if f.details:
        lines.append(f"Details from the jeweller (true, you may use them): {f.details}")
    lines += [
        f"Main search keyword, use it word for word in the first sentence: \"{keys[0]}\".",
        *([f"Also work in naturally if they fit: {', '.join(keys[1:4])}."] if p.limit > 500 else []),
        f"Tone: {tone}. Goal: {goal}.",
    ]
    if occ:
        lines.append(f"Timely angle: {occ[0].angle}.")
    if b.city:
        lines.append(f"The shop is in {b.city}.")
    if b.call():
        lines.append(f"End with this call to action: {b.call()}")
    lines.append(f"Platform style: {p.guide}")
    if p.title:
        lines.append(f"Answer as two lines: 'TITLE: ...' (under {min(p.title, 70)} characters, starting with the "
                     f"main keyword) then 'TEXT: ...'.")
    lines += [
        "Rules: only say what is listed above. Do not invent prices, discounts, carats, certificates, "
        "materials or stock. No hashtags. No quotation marks.",
        "Use 1 to 3 fitting emojis." if b.emojis and p.key not in ("linkedin", "google") else "No emojis.",
        "Write in English" + (" (American spelling: jewelry)." if b.market == "global" else " (spelling: jewellery)."),
    ]
    if extra:
        lines.append(extra)
    lines.append("Answer with the post only.")
    return "\n".join(lines)


def _split_title(text: str, p: Platform) -> tuple[str, str]:
    if not p.title:
        return "", text
    m = re.search(r"(?im)^\s*title\s*:\s*(.+)$", text)
    title = m.group(1).strip() if m else ""
    body = re.sub(r"(?im)^\s*title\s*:.*$", "", text)
    body = re.sub(r"(?im)^\s*(text|description)\s*:\s*", "", body).strip()
    if not title:
        first, _, rest = body.partition("\n")
        if len(first) <= p.title and rest.strip():
            title, body = first.strip(), rest.strip()
    return title.strip(' "'), body


def _fit(text: str, room: int) -> str:
    """Cut at a sentence end (else a word end) so the text fits in `room` characters."""
    if len(text) <= room:
        return text
    cut = text[:room]
    ends = [cut.rfind(c) for c in (".", "!", "?", "\n")]
    end = max(ends)
    if end >= room * 0.5:
        return cut[:end + 1].rstrip()
    return cut[:cut.rfind(" ")].rstrip(" ,;:-") + "…" if " " in cut else cut[:room - 1] + "…"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def template_text(b: Brief, p: Platform, keys: list[str], variant: int = 0) -> str:
    """A ready-made caption, used when no writer is free. Built only from the facts."""
    f, occ = b.facts, b.occasions()
    kw = keys[0]
    e = b.emojis and p.key not in ("linkedin", "google")
    spark = " ✨" if e else ""
    traits = ", ".join(f.traits[:2])
    openers = {
        "elegant": [f"Meet our {kw}{spark}", f"Quiet elegance: the {kw}{spark}"],
        "festive": [f"Celebrate in style with this {kw}{spark}", f"Festive sparkle, starring our {kw}{spark}"],
        "romantic": [f"Say it with our {kw}{' 💛' if e else ''}", f"A {kw} made for the one you love{' 💛' if e else ''}"],
        "playful": [f"Okay, this {kw} is our new obsession{spark}", f"Plot twist: your new favourite is this {kw}{spark}"],
        "minimal": [f"The {kw}. Simple, modern, yours.", f"Less, but better: our {kw}."],
        "luxury": [f"Make an entrance with this {kw}{spark}", f"Statement-making luxury: our {kw}{spark}"],
        "professional": [f"Introducing our {kw}.", f"A closer look at our {kw}."],
    }
    first = openers.get(b.tone, openers["elegant"])[variant % 2]
    body = []
    if traits:
        body.append(f"Designed with {traits} details.")
    if f.details:
        body.append("The details: " + f.details.strip().rstrip(".") + ".")
    if occ:
        body.append(f"Perfect for {occ[0].label}{' and every celebration after' if p.key != 'x' else ''}.")
    elif p.key != "x":
        body.append("Made to be worn and loved every day.")
    if p.key == "facebook":
        body.append("Which occasion would you wear it to?")
    if p.key == "linkedin":
        body.append(f"Every line of this {f.kind} is designed to catch the light and sit comfortably.")
    if p.key == "google" and b.city:
        body.append(f"See it at our store in {b.city}.")
    cta = b.call()
    if p.key == "x":
        return " ".join([first if first[-1:] in ".!?✨💛" else first + "."] + body[:1] + ([cta] if cta else []))
    if p.title:
        title = _cap(f"{kw}" + (f" for {occ[0].label}" if occ else f" · {spell('fine jewellery', b.market)}"))
        lead = first if first[-1:] in ".!?✨💛" else first + "."
        return f"TITLE: {title}\nTEXT: {' '.join([lead] + body + ([cta] if cta else []))}"
    sep = "\n\n" if p.key in ("instagram", "linkedin") else " "
    lead = first if sep != " " or first[-1:] in ".!?✨💛" else first + "."
    return sep.join([lead] + [" ".join(body)] + ([cta] if cta else []))


def house_spelling(text: str, market: str) -> str:
    if market == "global":
        return spell(text, market)
    return re.sub(r"\b([Jj])ewelry\b", lambda m: m.group(1) + "ewellery", text)


def finish(b: Brief, p: Platform, raw: str, keys: list[str], tags: list[str]) -> dict:
    """Writer answer -> a post that fits the platform: no invented claims, keyword up front,
    our hashtags at the end, within the length limit."""
    title, body = _split_title(house_spelling(_clean_answer(raw), b.market), p)
    body, dropped = scrub(body, b.facts.details + " " + b.call())
    title, dropped_t = scrub(title, b.facts.details)
    dropped += dropped_t
    kw = keys[0]
    occ = b.occasions()
    if p.title and (not title or title.lower() == kw.lower()):   # a bare keyword is a weak title
        title = _cap(kw) + (f" for {occ[0].label}" if occ else f" · {spell('Fine jewellery', b.market)}")
    title = _cap(title)
    if kw.lower() not in (title + " " + body).lower():
        if p.title and len(f"{_cap(kw)} | {title}") <= p.title:
            title = f"{_cap(kw)} | {title}" if title else _cap(kw)
        else:
            body = f"{_cap(kw)}. {body}" if p.key in ("x", "threads") else f"{_cap(kw)}{' ✨' if b.emojis and p.key not in ('linkedin', 'google') else ''}\n\n{body}"
    cta = b.call()
    if cta and cta.lower()[:18] not in body.lower():
        body = body.rstrip() + ("\n\n" if p.key in ("instagram", "linkedin") else " ") + cta
    tag_line = " ".join(tags)
    room = p.limit - (len(tag_line) + 2 if tags else 0)
    body = _fit(body.strip(), room)
    title = _fit(title, p.title) if p.title else ""
    text = body + (("\n\n" if p.key != "x" else " ") + tag_line if tags else "")
    first = body.split("\n", 1)[0]
    return {
        "platform": p.key, "label": p.label, "title": title, "text": text, "hashtags": tags,
        "chars": len(text), "limit": p.limit, "title_limit": p.title,
        "hook_ok": (len(first) <= p.hook) if p.hook else None, "hook": p.hook,
        "keyword_in_start": kw.lower() in (title + " " + body[:160]).lower(),
        "dropped": dropped,
    }


# ---------- writers ----------

_pollinations_off_until = 0.0


def pollinations_writer():
    """Pollinations' text model (it looks at the picture). None when not connected or
    out of balance (checked again after 15 minutes)."""
    if "pollinations" not in sketch.connected() or time.time() < _pollinations_off_until:
        return None

    def write(prompt: str, picture: bytes | None, max_tokens: int) -> str:
        global _pollinations_off_until
        try:
            return sketch.call_pollinations_text(prompt, picture, max_tokens)
        except SketchError as e:
            if e.reason == "balance" or e.status in (401, 402, 403):
                _pollinations_off_until = time.time() + 900
            raise
    return write


def local_writer(engine):
    """The local Qwen3-1.7B (the jewellery judge, already loaded). None if absent; raises
    SketchError while a picture is being drawn (the model lets go of its memory then)."""
    ask = getattr(getattr(engine, "domain", None), "_judge", None)
    ask = getattr(ask, "__wrapped__", ask)
    judge = getattr(ask, "__self__", None)
    if judge is None or not hasattr(judge, "tok"):
        return None

    def write(prompt: str, picture: bytes | None, max_tokens: int) -> str:
        if hasattr(judge, "available") and not judge.available():
            raise SketchError("local writer is away", 503)
        msgs = [{"role": "user", "content": prompt}]
        text = judge.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = judge.tok(text, return_tensors="pt").to(judge.dev)
        with judge.gpu, judge.torch.no_grad():
            model = judge.model
            if model is None:
                raise SketchError("local writer is away", 503)
            out = model.generate(**x, max_new_tokens=max_tokens, do_sample=True, temperature=0.8, top_p=0.92,
                                 repetition_penalty=1.1)
        return judge.tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
    return write


def write_post(b: Brief, key: str, picture: bytes | None, writers: list[tuple[str, object]],
               variant: int = 0) -> dict:
    """One platform's post: the first writer that answers usefully, else the template."""
    p = PLATFORMS[key]
    occ = b.occasions()
    keys = keywords(b.facts, occ, b.city, b.market)
    tags = pick_tags(hashtags(b.facts, occ, b.brand, b.city, b.market), p.tags)
    extra = "Write a different version from before: new opening line and wording." if variant else ""
    for name, write in writers:
        try:
            raw = write(prompt_for(b, p, keys, extra), picture if name == "ai" else None, p.tokens)
        except SketchError as e:
            print(f"social writer {name}: {e.message}", flush=True)
            continue
        except Exception as e:
            print(f"social writer {name}: {e!r}", flush=True)
            continue
        post = finish(b, p, raw, keys, tags)
        if len(re.findall(r"[A-Za-z]{2,}", _clean_answer(raw))) >= 6:   # a real answer, not an empty or broken one
            return {**post, "writer": name, "keywords": keys}
    post = finish(b, p, template_text(b, p, keys, variant), keys, tags)
    return {**post, "writer": "template", "keywords": keys}


def writers_for(engine, writer=None) -> list[tuple[str, object]]:
    if writer is not None:
        return [("ai", writer)]
    out = []
    w = pollinations_writer()
    if w:
        out.append(("ai", w))
    w = local_writer(engine)
    if w:
        out.append(("local", w))
    return out


# ---------- background jobs (the page polls; a full set takes ~1 minute on the local writer) ----------

WRITE_LOCK = threading.Lock()      # one set written at a time (shares the GPU with search)


class Jobs:
    def __init__(self, engine=None, writer=None):
        self.engine, self.writer = engine, writer
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def _sweep(self):
        cut = time.time() - 3600
        for k in [k for k, j in self.jobs.items() if j["ts"] < cut and j["status"] != "running"]:
            del self.jobs[k]
        while len(self.jobs) > 200:   # keep memory small: oldest first
            del self.jobs[min(self.jobs, key=lambda k: self.jobs[k]["ts"])]

    def start(self, uid: str, picture: bytes | None, brief: Brief) -> dict:
        job_id = secrets.token_urlsafe(9)
        with self.lock:
            self._sweep()
            self.jobs[job_id] = {"id": job_id, "uid": uid, "ts": time.time(), "status": "running", "stage": "reading",
                                 "posts": {}, "order": list(brief.platforms), "facts": None, "trends": [],
                                 "keywords": [], "error": None, "_picture": picture, "_brief": brief}
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return self.get(job_id, uid)

    def _set(self, job_id, **kw):
        with self.lock:
            self.jobs[job_id].update(kw)

    def _run(self, job_id):
        j = self.jobs[job_id]
        b, picture = j["_brief"], j["_picture"]
        try:
            b.facts = read_facts(self.engine, picture, b.facts.details)
            occ = b.occasions()
            self._set(job_id, stage="writing", facts=public_facts(b.facts), trends=[public_occasion(o) for o in occ],
                      keywords=keywords(b.facts, occ, b.city, b.market),
                      hashtag_pool=hashtags(b.facts, occ, b.brand, b.city, b.market))
            writers = writers_for(self.engine, self.writer)
            with WRITE_LOCK:
                for key in b.platforms:
                    post = write_post(b, key, picture, writers)
                    with self.lock:
                        j["posts"][key] = post
            self._set(job_id, status="done", stage="done")
        except Exception as e:
            print(f"social job failed: {e!r}", flush=True)
            self._set(job_id, status="error", error="The captions could not be written. Try again.")

    def get(self, job_id: str, uid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            if not j or j["uid"] != uid:
                return None
            return {k: v for k, v in j.items() if k != "uid" and not k.startswith("_")}

    def rewrite(self, job_id: str, uid: str, platform: str, variant: int) -> dict:
        with self.lock:
            j = self.jobs.get(job_id)
            if not j or j["uid"] != uid:
                raise SketchError("This post set has expired. Make the captions again.", 404)
            if j["status"] == "running":
                raise SketchError("Wait until all captions are written.", 409)
            b, picture = j["_brief"], j["_picture"]
        if platform not in PLATFORMS:
            raise SketchError("Unknown platform.")
        with WRITE_LOCK:
            post = write_post(b, platform, picture, writers_for(self.engine, self.writer), variant=max(1, variant))
        with self.lock:
            j["posts"][platform] = post
            if platform not in j["order"]:
                j["order"].append(platform)
        return post


def public_facts(f: Facts) -> dict:
    return {"kind": f.kind, "metal": f.metal, "stone": f.stone, "stone_style": f.stone_style,
            "traits": f.traits, "keyword": f.keyword(), "sure": f.sure}


def public_occasion(o: Occasion) -> dict:
    return {"key": o.key, "label": o.label, "day": o.day.isoformat(),
            "start": o.start.isoformat() if o.start else None}


def make_brief(platforms: list[str], details: str = "", **kw) -> Brief:
    plats = [p for p in dict.fromkeys(platforms) if p in PLATFORMS]
    if not plats:
        raise SketchError("Choose at least one platform.")
    return Brief(facts=Facts(details=" ".join(details.split())[:DETAILS_MAX]), platforms=plats, **kw)


def options(today: dt.date | None = None) -> dict:
    return {
        "platforms": [{"key": p.key, "label": p.label, "limit": p.limit, "tags": p.tags, "title": p.title}
                      for p in PLATFORMS.values()],
        "tones": [{"key": k, "label": v[0]} for k, v in TONES.items()],
        "goals": [{"key": k, "label": v[0]} for k, v in GOALS.items()],
        "ctas": [{"key": k, "label": v or "No call to action"} for k, v in CTAS.items()],
        "markets": [{"key": k, "label": v} for k, v in MARKETS.items()],
        "trending": {m: [public_occasion(o) for o in trending(today, m)] for m in MARKETS},
        "occasions": [public_occasion(o) for o in OCCASIONS if o.day >= (today or dt.date.today())],
    }
