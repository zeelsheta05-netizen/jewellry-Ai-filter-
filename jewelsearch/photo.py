"""A shopper's photo: read it safely and find the piece in it.

Plain image processing, no model. The search engine embeds the crops made
here and reads the design DNA from them (`SearchEngine.read_photo`).

Catalogue crops (scripts/make_crops.py) are the piece's bounding box padded
6% to a square. A photo is cropped the same way when the piece stands out from
its background: on simulated shopper photos that raised "design found on the
first page" from 69% to 79% (search.py has the full numbers). Whitening the
background around the piece was also tried and did not help.
"""
import base64
import binascii
import io

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

MAX_BYTES = 8 * 1024 * 1024    # decoded upload; the page sends a ~1024px JPEG of a few hundred KB
MAX_PIXELS = 50_000_000        # checked from the header, before decoding (decompression bombs)
MIN_SIDE = 64
WORK_SIDE = 1024               # longest side kept for analysis
PAD = 0.06                     # same padding as the catalogue crops
FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "MPO"}   # MPO: some phone cameras' JPEGs

SMALL = 192                    # side for the background analysis
EDGE = 0.04                    # border strip that samples the background
MIN_UNIFORM = 0.6              # share of the border that must fit the background model
MIN_FG, MAX_FG = 0.003, 0.7    # foreground share of the frame for a usable cut-out
MAX_BOX = 0.8                  # share of the frame the piece's box may cover
MAX_EDGE_FG = 0.2              # a cut-out touching more of the border than this failed (or the piece is cut off)
SHADOW_CHROMA = 0.035          # a shadow keeps the background's colour (chromaticity) ...
SHADOW_DARK = (0.35, 0.93)     # ... at this fraction of its brightness


class PhotoError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status, self.message = status, message


def from_data_url(s: str) -> bytes:
    head, _, body = s.partition(",")
    if not (head.startswith("data:image/") and head.endswith(";base64")):
        raise PhotoError("Send the photo as an image file.")
    try:
        data = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        raise PhotoError("The photo could not be read. Try another file.")
    if len(data) > MAX_BYTES:
        raise PhotoError("The photo is too large. Use one under 8 MB.", 413)
    return data


def read(data: bytes) -> Image.Image:
    """Bytes -> upright RGB photo, longest side at most WORK_SIDE."""
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in FORMATS:
            raise PhotoError("Use a JPG, PNG or WebP photo.")
        if im.width * im.height > MAX_PIXELS:
            raise PhotoError("The photo has too many pixels. Use a smaller one.", 413)
        im.draft("RGB", (WORK_SIDE, WORK_SIDE))   # JPEG: decode at a reduced size, much faster
        im = ImageOps.exif_transpose(im)          # phone photos are often stored sideways
        im.load()
    except PhotoError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise PhotoError("This file isn't a photo that can be read. Use a JPG, PNG or WebP image.")
    if min(im.size) < MIN_SIDE:
        raise PhotoError(f"The photo is too small. Use one at least {MIN_SIDE} pixels on each side.")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")   # a cut-out: put it on white, like the catalogue
        im = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im.alpha_composite(rgba)
    im = im.convert("RGB")
    im.thumbnail((WORK_SIDE, WORK_SIDE), Image.LANCZOS)
    return im


def _moving_median(edge: np.ndarray, win: int) -> np.ndarray:
    """Median along a border line, so a piece touching the edge doesn't paint the background."""
    if win < 3 or len(edge) <= win:
        return edge
    pad = win // 2
    padded = np.concatenate([edge[:1].repeat(pad, 0), edge, edge[-1:].repeat(pad, 0)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, win, axis=0)   # (n, 3, win)
    return np.median(windows, axis=-1)


def background(a: np.ndarray):
    """Smooth background model from the four borders (handles gradients and
    vignetting): the blend of the top-bottom and left-right interpolations.
    -> (model HxWx3, border distances to it)"""
    h, w = a.shape[:2]
    b = max(2, round(EDGE * min(h, w)))
    top, bottom = np.median(a[:b], 0), np.median(a[-b:], 0)                # (w, 3)
    left, right = np.median(a[:, :b], 1), np.median(a[:, -b:], 1)          # (h, 3)
    top, bottom = (_moving_median(e, max(3, w // 6) | 1) for e in (top, bottom))
    left, right = (_moving_median(e, max(3, h // 6) | 1) for e in (left, right))
    ty = np.linspace(0, 1, h)[:, None, None]
    tx = np.linspace(0, 1, w)[None, :, None]
    model = 0.5 * ((1 - ty) * top[None] + ty * bottom[None]) + 0.5 * ((1 - tx) * left[:, None] + tx * right[:, None])
    dist = np.linalg.norm(a - model, axis=-1)
    border = np.concatenate([dist[:b].ravel(), dist[-b:].ravel(), dist[:, :b].ravel(), dist[:, -b:].ravel()])
    return model, border


def piece_mask(im: Image.Image):
    """Foreground mask at SMALL size, or None when the background is busy
    (a face, a room) or nothing stands out from it."""
    small = im.copy()
    small.thumbnail((SMALL, SMALL), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    model, border = background(a)
    tol = float(np.clip(3.5 * np.percentile(border, 75), 18, 60))
    if (border < tol).mean() < MIN_UNIFORM:
        return None
    dist = np.linalg.norm(a - model, axis=-1)
    fg = dist > tol
    # a cast shadow is the background, only darker: same chromaticity, lower brightness
    lum, blum = a.sum(-1) + 1, model.sum(-1) + 1
    chroma = np.abs(a / lum[..., None] - model / blum[..., None]).sum(-1)
    ratio = lum / blum
    fg &= ~((chroma < SHADOW_CHROMA) & (ratio > SHADOW_DARK[0]) & (ratio < SHADOW_DARK[1]))
    m = Image.fromarray((fg * 255).astype(np.uint8))
    # close small gaps (between stones), then drop isolated specks (noise, dust)
    m = m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    mask = np.asarray(m) > 0
    if not (MIN_FG <= mask.mean() <= MAX_FG):
        return None
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if edge.mean() > MAX_EDGE_FG:
        return None
    # specks scattered over the whole frame (a textured background) are not a piece;
    # and a piece filling the frame needs no cut-out
    ys, xs = np.nonzero(mask)
    (y0, y1), (x0, x1) = np.quantile(ys, [0.003, 0.997]), np.quantile(xs, [0.003, 0.997])
    if (y1 - y0) * (x1 - x0) > MAX_BOX * mask.size:
        return None
    return mask


def piece_box(im: Image.Image, mask: np.ndarray):
    """Square box around the piece in `im` pixels, padded like the catalogue
    crops. Coordinate quantiles, not min/max, so a stray speck can't stretch it."""
    ys, xs = np.nonzero(mask)
    sy, sx = im.height / mask.shape[0], im.width / mask.shape[1]
    y0, y1 = np.quantile(ys, [0.003, 0.997])
    x0, x1 = np.quantile(xs, [0.003, 0.997])
    y0, y1, x0, x1 = y0 * sy, (y1 + 1) * sy, x0 * sx, (x1 + 1) * sx
    side = max(y1 - y0, x1 - x0) * (1 + 2 * PAD)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    return (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))


def square_crop(im: Image.Image, box, fill=(255, 255, 255)) -> Image.Image:
    """Crop that may reach outside the photo; the outside is filled."""
    side = box[2] - box[0]
    out = Image.new("RGB", (side, side), fill)
    out.paste(im.crop(box), (0, 0))
    if any(v < 0 for v in box[:2]) or box[2] > im.width or box[3] > im.height:
        # crop() fills outside pixels with black: repaint them
        x0, y0 = max(0, -box[0]), max(0, -box[1])
        x1, y1 = min(side, im.width - box[0]), min(side, im.height - box[1])
        inside = Image.new("L", (side, side), 0)
        inside.paste(255, (x0, y0, x1, y1))
        out = Image.composite(out, Image.new("RGB", (side, side), fill), inside)
    return out


def square_full(im: Image.Image):
    """The whole photo padded to a square on its own border colour
    -> (square image, its box in photo pixels, the fill colour)."""
    side = max(im.size)
    a = np.asarray(im, dtype=np.float32)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(edge, 0))
    box = ((im.width - side) // 2, (im.height - side) // 2, (im.width - side) // 2 + side, (im.height - side) // 2 + side)
    return square_crop(im, box, fill), box, fill


def heat_box(heat: np.ndarray, threshold: float, keep: float = 0.25):
    """Grid cells (rows x cols) of how likely each patch of the square photo is jewellery
    -> (r0, c0, r1, c1) around the connected groups of cells at least `threshold` sure
    (end exclusive), or None when no cell is. Every group holding at least `keep` of the
    largest group's certainty is included: both earrings of a pair, a pendant and its
    chain; a stray patch elsewhere is not."""
    on = heat >= threshold
    if not on.any():
        return None
    seen = np.zeros_like(on)
    groups = []
    rows, cols = on.shape
    for r, c in zip(*np.nonzero(on)):
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:   # 8-connected: a thin ring band runs diagonally
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and on[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys, xs = zip(*cells)
        groups.append((float(heat[list(ys), list(xs)].sum()), min(ys), min(xs), max(ys) + 1, max(xs) + 1))
    top = max(g[0] for g in groups)
    kept = [g for g in groups if g[0] >= keep * top]
    return (min(g[1] for g in kept), min(g[2] for g in kept), max(g[3] for g in kept), max(g[4] for g in kept))


def heat_on_mask(heat: np.ndarray, square_box, size: tuple, mask: np.ndarray) -> float:
    """Mean of the piece finder's map over a cut-out mask (piece_mask, analysis size): how much
    of what the background cut-out found is jewellery."""
    side = square_box[2] - square_box[0]
    h = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)
    h = h.crop((-square_box[0], -square_box[1], -square_box[0] + size[0], -square_box[1] + size[1]))
    h = np.asarray(h.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR), np.float32) / 255
    return float(h[mask].mean()) if mask.any() else 0.0


def grid_to_photo(cells, grid: int, square_box) -> tuple:
    """heat_box cells -> a square box in photo pixels around them, padded like the catalogue crops."""
    r0, c0, r1, c1 = cells
    cell = (square_box[2] - square_box[0]) / grid
    x0, y0 = square_box[0] + c0 * cell, square_box[1] + r0 * cell
    x1, y1 = square_box[0] + c1 * cell, square_box[1] + r1 * cell
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))


def views(im: Image.Image) -> dict:
    """The crops of one photo:
      full      the whole photo, padded to a square on its own background colour
      piece     square crop around the piece, padded like the catalogue (when it stands out)
    plus "mask" (the piece, at analysis size), "box": the piece's box as
    fractions of the photo, for the page to outline, and "fill": the photo's border colour."""
    out = {}
    out["full"], _, fill = square_full(im)
    out["fill"] = fill
    mask = piece_mask(im)
    out["mask"] = mask
    out["box"] = None
    if mask is not None:
        box = piece_box(im, mask)
        out["piece"] = square_crop(im, box, fill)
        out["box"] = [round(max(0, box[0]) / im.width, 4), round(max(0, box[1]) / im.height, 4),
                      round(min(im.width, box[2]) / im.width, 4), round(min(im.height, box[3]) / im.height, 4)]
    return out

