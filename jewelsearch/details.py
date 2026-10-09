"""Design details read by a vision-language model: the properties that make a
design itself, which the image embeddings blur together (every halo ring
looks alike to them).

Qwen3-VL-2B-Instruct (Apache 2.0) runs on this Mac and answers yes / no
questions about a picture; the probability of "yes" is read from one forward
pass (no text is generated), all questions of a picture in one batch.

Only questions it answered consistently are asked. First tried on a shop photo
and 9 catalogue rings checked by eye; "twisted band" and "split / open band"
failed there (it called plain bands twisted). The check on the whole collection
(see SHOWN) found that only the diamond-set band means what its name says.

The same reader reads every design's front view once
(scripts/build_details_index.py -> data/index/details.npy), so a photo's
details can be matched with the designs' own.
"""
import numpy as np
import torch
from PIL import Image

from . import memory
from .config import GPU, device
from .embedder import on_white

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SIDE = 448            # picture size the model sees (longest side)
# key -> (question, label shown when the answer is yes, matched with the collection?)
QUESTIONS = {
    "halo": ("Is the centre stone surrounded by a halo, a ring of small diamonds around it?", "Halo around the centre stone", True),
    "clusters": ("Are there separate small clusters of diamonds on the band, away from the centre?",
                 "Diamond clusters on the band", True),
    "pave_band": ("Is the band covered with a row of small diamonds?", "Diamond-set band", True),
    "two_tone": ("Is the piece made in two different metal colours (two-tone)?", "Two-tone metal", False),
}
# Checked on the finished index (2026-10-05) the names don't all hold: the reader says "halo" and
# "clusters on the band" for most rings with any small diamonds (halo for 70% of the rings whose job
# card has no stone standing out; by eye on 30 random rings, halo right for about 7 of 23 "yes",
# clusters about 4 of 21), while "diamond-set band" was right for about 17 of 18. Its readings are
# consistent, though: a photo and the design's own render agree 92-95% (300 simulated photos), and a
# shop's worn photo and its studio shot 95-100% (Svaraa, 188 photos). So all three are matched with
# the collection (DETAILS_WEIGHT in search.py) and only the band is shown as a label. Two-tone is not
# asked any more: the worn and studio photos of the same product agreed only 65% of the time.
SHOWN = {"pave_band"}
YES = 0.8             # shown / used as a property when the reading is at least this sure...
NO = 0.2              # ...and as its absence at most this
LETTERS = "ABCDEFGHIJKL"


class DetailReader:
    def __init__(self, model_id: str = MODEL):
        import threading
        from transformers import AutoProcessor
        self.model_id = model_id
        self.dev = device()
        self.dtype = torch.float16 if self.dev in ("mps", "cuda") else torch.float32
        self._load_lock = threading.Lock()
        self.model = None
        try:
            self.proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except OSError:
            self.proc = AutoProcessor.from_pretrained(model_id)
        self.load()
        tok = self.proc.tokenizer
        self.proc.tokenizer.padding_side = "left"   # the answer is read at the last position of every row
        ids = lambda w: tok.encode(w, add_special_tokens=False)[0]
        self.yes, self.no = [ids("yes"), ids("Yes")], [ids("no"), ids("No")]
        self.keys = [k for k, (_, _, matched) in QUESTIONS.items() if matched]
        self.letters = [ids(c) for c in LETTERS]
        memory.register(self)   # lets go of its ~4.5 GB while the local picture model draws

    def load(self) -> bool:
        """Load the weights (at start, and again after a drawing). True when loaded now."""
        with self._load_lock:
            if self.model is not None:
                return False
            from transformers import Qwen3VLForConditionalGeneration
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype=self.dtype, local_files_only=True)
            except OSError:
                model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype=self.dtype)
            with GPU:
                self.model = model.to(self.dev).eval()
            return True

    def release(self) -> bool:
        """Let go of the weights (the local picture model is about to draw)."""
        with self._load_lock, GPU:   # waits for a reading in progress
            if self.model is None:
                return False
            self.model = None
            return True

    def ready(self) -> bool:
        """Loaded, or loaded now; False while a picture is being drawn (the reading is skipped then)."""
        if self.model is not None:
            return True
        if memory.DRAWING.is_set():
            return False
        self.load()
        return self.model is not None

    @torch.no_grad()
    def read(self, im: Image.Image, keys: list[str] | None = None, side: int = SIDE) -> dict[str, float]:
        """picture -> {question key: probability of yes}; {} while a picture is being drawn."""
        keys = keys or self.keys
        if not self.ready():
            return {}
        im = on_white(im).convert("RGB")
        im.thumbnail((side, side))
        convs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                               {"type": "text", "text": QUESTIONS[k][0] + " Answer yes or no."}]}]
                 for k in keys]
        x = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt", processor_kwargs={"padding": True})
        with GPU:
            model = self.model
            if model is None:   # let go in between (a drawing started)
                return {}
            logits = model(**x.to(self.dev)).logits[:, -1].float()
            yes = torch.logsumexp(logits[:, self.yes], dim=1)
            no = torch.logsumexp(logits[:, self.no], dim=1)
            p = torch.sigmoid(yes - no).cpu().numpy()
        return {k: float(v) for k, v in zip(keys, p)}


    @torch.no_grad()
    def choose(self, im: Image.Image, questions: list[tuple[str, list[str]]], side: int = SIDE) -> list[list[float]]:
        """Multiple-choice questions about one picture, all in one batch:
        [(question, [option, ...]), ...] -> per question the probability of each option,
        read from the answer letter's logits (no text is generated).
        Raises RuntimeError while a picture is being drawn (callers fall back)."""
        if not self.ready():
            raise RuntimeError("details reader is let go while a picture is drawn")
        im = on_white(im).convert("RGB")
        im.thumbnail((side, side))
        convs = []
        for q, opts in questions:
            listed = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
            convs.append([{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": f"{q}\n{listed}\nAnswer with the letter only."}]}])
        x = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt", processor_kwargs={"padding": True})
        with GPU:
            model = self.model
            if model is None:
                raise RuntimeError("details reader is let go while a picture is drawn")
            logits = model(**x.to(self.dev)).logits[:, -1].float()
        out = []
        for row, (_, opts) in zip(logits, questions):
            pick = row[torch.tensor(self.letters[:len(opts)], device=row.device)]
            out.append(torch.softmax(pick, 0).cpu().tolist())
        return out


def shown(reading: dict[str, float]) -> list[dict]:
    """The details to show for a picture: the trusted ones (SHOWN) read as present."""
    return [{"key": k, "label": QUESTIONS[k][1], "p": round(p, 3)} for k, p in reading.items()
            if k in SHOWN and p >= YES]


def agreement(reading: dict[str, float], designs: np.ndarray, keys: list[str]) -> np.ndarray | None:
    """How many of the photo's clear details (present or absent) each design shares,
    from the designs' own readings (n_designs, n_keys). None when nothing is clear."""
    score, used = np.zeros(len(designs)), 0
    for j, k in enumerate(keys):
        if k not in reading or not QUESTIONS[k][2]:
            continue
        p = reading[k]
        if p >= YES:
            score += designs[:, j]
        elif p <= NO:
            score += 1 - designs[:, j]
        else:
            continue
        used += 1
    return score / used if used else None
