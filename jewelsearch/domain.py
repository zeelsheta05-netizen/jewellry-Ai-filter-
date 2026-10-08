"""Keep the search to jewellery. A prompt about anything else ("pizza", "weather
today", "boxing ring", "wedding dress") gets "Can't find designs like …" and
no results, instead of 8 designs that happen to be the least unlike it.

Three layers, cheapest first, measured with scripts/eval_domain.py:

1. The parser's own words (jewelsearch/query.py). A prompt with no
   jewellery word the parser knows is refused. A prompt made only of words it
   knows is accepted. About 3 in 4 real prompts end here, with no model call.
   Gujarati and Hindi words the parser doesn't know count as filler ("જોઈએ",
   "हो"): the judge below reads those scripts poorly.
2. The judge: a small local language model (Qwen3-1.7B, Apache 2.0) reads
   prompts that mix known words with unknown English / Hinglish ones and
   answers yes (jewellery), other (a kind of jewellery the catalogue doesn't
   have: anklet, toe ring...) or no. About 0.7 s per new prompt, cached.
3. The picture check, for prompts the judge let through: the image model's
   calibrated match score of the whole prompt against every design must not
   fall far below a plain "a photo of a <piece>". It catches the judge's
   mistakes on look-alikes ("mehndi design for bride", "necklace hanger stand").

Budgets ("under 30000", "50k") are taken out first and noted: prices aren't
in the catalogue yet, which is no reason to refuse the design.
"""
import logging
import re
import threading
from dataclasses import dataclass
from functools import lru_cache

from .query import parse

log = logging.getLogger(__name__)

JUDGE_MODEL = "Qwen/Qwen3-1.7B"
# Lowest picture-check score accepted (calibrated logit of the prompt minus that of
# "a photo of a <piece>", best design each). Measured on prompts that reach the judge:
# real prompts -2.6 at the lowest in the tuning set (-4.5 in a held-out set), prompts
# the judge wrongly let through -4.3 to -8.8. -5 keeps a margin for real prompts.
PICTURE_MIN = -5.0

JUDGE_SYSTEM = """You screen searches typed into the search box of a jewellery shop's design catalogue (gold rings, earrings, pendants, necklaces and bracelets, most with diamonds). Searches may be in English, Hindi, Gujarati or Hinglish.

Decide what the shopper wants to see, and answer with exactly one word:
yes - jewellery: a ring, earrings, a pendant, a necklace, a bracelet, bangles, a chain or jewellery in general, of any design, motif, shape, size or style. When the search names such a piece as the thing wanted, the answer is yes whatever else it mentions: the occasion (wedding, reception, party, festival, office), the person, the budget, the look or an outfit to match.
other - a kind of jewellery this catalogue does not have (anklet, nose pin, toe ring, brooch, maang tikka...).
no - something that is not jewellery: clothes, shoes, bags, make-up, flowers, cakes, food, decorations, venues, services, phones, vehicles or other products; prices or rates; information, how-to, news, sport, films, songs or chat. Words like ring, chain, diamond, star, stone, heart or gold used for something else (a boxing ring, a diamond painting, a star hotel) are no.

Examples:
a ring that looks like a lotus -> yes
heavy necklace for my sister's sangeet -> yes
earrings with a swan -> yes
something sparkly for diwali -> yes
maang tikka -> other
brooch for a suit -> other
dupatta for wedding -> no
car keyring -> no
ring tone download -> no
diamond price per carat -> no
sangeet dance songs -> no
rose plant -> no"""
VERDICTS = ("yes", "other", "no")

# a budget: "under 30000", "below 50k", "rs 20,000", "25000 rs", "₹15000"; never a gold purity ("18k", "22k gold")
PRICE = re.compile(r"(?:under|below|within|less than|upto|up to|budget|rs\.?|inr|₹)\s*\d[\d,]*\s*k?\b"
                   r"|(?<![\d,])(?!(?:9|10|14|18|22|24)\s*k\b)\d[\d,]*\s*(?:k|rs|rupees|inr|₹)(?![a-z])", re.I)
PRICE_WORDS = re.compile(r"\b(?:price|prices|rate|rates|cost|bhav|kimat|keemat)\b|ભાવ|કિંમત|कीमत|भाव|दाम", re.I)
ENGLISH_NOUN = {"ring": "ring", "earrings": "earrings", "pendant": "pendant", "necklace": "necklace",
                "bracelet": "bracelet"}
HINT = ("Design Finder searches only this jewellery collection. Describe a ring, earrings, a pendant, "
        "a necklace or a bracelet, for example “rose gold ring for my mother”.")
HINTS = {
    "other_type": "The collection has rings, earrings, pendants, necklaces and bracelets only.",
    "price": "Design Finder shows the collection's designs, not prices or rates.",
}


@dataclass
class Verdict:
    ok: bool
    text: str                 # the prompt to search with (a budget taken out)
    reason: str = ""          # why it was refused: nothing | not_jewellery | other_type | no_match
    via: str = "words"        # what decided: words | judge | picture
    budget: str | None = None  # the budget taken out, if any

    def refusal(self, prompt: str) -> dict:
        shown = " ".join(prompt.split())
        shown = shown if len(shown) <= 60 else shown[:57].rstrip() + "…"
        hint = HINTS["price"] if PRICE_WORDS.search(prompt) else HINTS.get(self.reason, HINT)
        return {"reason": self.reason, "title": f"Can't find designs like “{shown}”", "hint": hint}


def strip_budget(prompt: str) -> tuple[str, str | None]:
    m = PRICE.search(prompt)
    if not m:
        return prompt, None
    clean = " ".join(PRICE.sub(" ", prompt).split())
    return (clean or prompt), " ".join(m.group().split())


def understood(q) -> bool:
    return bool(q.categories or q.metal or q.intents or q.not_intents or q.shape or q.exclude_categories
                or q.exclude_metals)


def english(prompt: str) -> str:
    """Piece names the parser knows in Hindi / Gujarati / Hinglish ("haar",
    "bali") in English for the judge, which doesn't know them."""
    out = prompt
    for term, kind, value, _ in parse(prompt).terms:
        if kind == "category" and ENGLISH_NOUN[value] not in term.lower():
            out = re.sub(re.escape(term), ENGLISH_NOUN[value], out, count=1, flags=re.I)
    return out


class Judge:
    """Qwen3-1.7B on this Mac: one forward pass per prompt, the answer is the
    likeliest of the three verdict words (no free text is generated)."""

    def __init__(self, model_id: str = JUDGE_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .config import GPU, device
        self.gpu = GPU
        self.torch = torch
        self.dev = device()
        dtype = torch.float16 if self.dev in ("mps", "cuda") else torch.float32
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, local_files_only=True)
        except OSError:   # not downloaded yet
            self.tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        with GPU:
            self.model = model.to(self.dev).eval()
        self.ids = [self.tok.encode(v, add_special_tokens=False)[0] for v in VERDICTS]
        self.ask = lru_cache(maxsize=4096)(self._ask)

    def _ask(self, prompt: str) -> str:
        msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = self.tok(text, return_tensors="pt")
        with self.gpu, self.torch.no_grad():
            logits = self.model(**x.to(self.dev)).logits[0, -1].float()
            return VERDICTS[int(logits[self.ids].argmax())]


class Domain:
    """Decides whether a prompt is a jewellery search (see the module notes)."""

    def __init__(self, engine, judge="auto"):
        self.engine = engine
        self._judge = judge            # "auto" = load on first need, None = no judge, or a callable
        self._load_lock = threading.Lock()
        self.check = lru_cache(maxsize=4096)(self._check)

    def load_judge(self):
        """Load the judge now (the server does this at start-up, so no shopper waits)."""
        with self._load_lock:
            if self._judge == "auto":
                try:
                    self._judge = Judge().ask
                except Exception as e:   # the picture check still runs without it
                    log.warning("judge not available (%s); using the picture check alone", e)
                    print(f"domain: judge not available ({e}); using the picture check alone", flush=True)
                    self._judge = None
        return self._judge

    def _check(self, prompt: str) -> Verdict:
        text, budget = strip_budget(" ".join(prompt.split()))
        q = parse(text)
        if not understood(q):
            return Verdict(False, text, "nothing", "words", budget)
        if not q.free_text():
            return Verdict(True, text, via="words", budget=budget)
        judge = self.load_judge() if self._judge == "auto" else self._judge
        if judge:
            v = judge(english(text))
            if v != "yes":
                return Verdict(False, text, "other_type" if v == "other" else "not_jewellery", "judge", budget)
        if self.engine.picture_score(english(text), q) < PICTURE_MIN:
            return Verdict(False, text, "no_match", "picture", budget)
        return Verdict(True, text, via="judge" if judge else "picture", budget=budget)
