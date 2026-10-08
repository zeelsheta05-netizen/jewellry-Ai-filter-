#!/usr/bin/env python3
"""Render the top-8 results of prompts as contact sheets for visual review.

Usage: .venv/bin/python scripts/eval_sheets.py OUT_DIR "prompt 1" "prompt 2" ...
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch.config import DATA  # noqa: E402
from jewelsearch.search import SearchEngine  # noqa: E402

T, BG = 220, (243, 239, 232)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    eng = SearchEngine()
    for n, prompt in enumerate(sys.argv[2:]):
        r = eng.search(prompt)
        q = r["query"]
        sheet = Image.new("RGB", (4 * T, 2 * (T + 30) + 44), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), prompt[:140], fill="black")
        d.text((6, 20), f"cat={q['category']} metal={q['metal']} want={q['intents']} not={q['not_intents']}"[:140],
               fill="gray")
        for i, c in enumerate(r["results"]):
            im = Image.open(DATA / c["thumb"].lstrip("/")).convert("RGBA")
            bg = Image.new("RGBA", im.size, BG + (255,))
            bg.alpha_composite(im)
            x, y = (i % 4) * T, 44 + (i // 4) * (T + 30)
            sheet.paste(bg.convert("RGB").resize((T - 6, T - 6)), (x + 3, y))
            d.text((x + 6, y + T - 4), f"{c['design_id']} ({c['category']})", fill="black")
            d.text((x + 6, y + T + 8), ", ".join(c["tags"])[:40], fill="gray")
        sheet.save(out / f"sheet{n}.png")
        print(out / f"sheet{n}.png", "|", prompt)


if __name__ == "__main__":
    main()
