"""DINOv2 image vectors for search by photo (and by link).

SigLIP2 (embedder.py) knows what a picture shows ("a halo ring"), which is how
words and pictures meet. DINOv2 (Meta, Apache 2.0, self-supervised) is much
better at telling apart near-identical designs: the exact shape of a head, a
band, a cluster. Photo search ranks with both. Measured on 200 simulated
shopper photos (scripts/eval_photo_search.py has the full engine numbers):

    design ranked first      SigLIP2 58%   DINOv2 75%   both 78.5%
    design on the first page          86%          93.5%        93.5%

Catalogue views: data/index/dino_views.npy, rows in the order of views.json
(scripts/build_dino_index.py). The search ignores a file made for another index,
model or size.
"""
import numpy as np
import torch
from PIL import Image

from .config import GPU, device
from .embedder import on_white

MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"   # timm name; weights from the Hugging Face hub
SIZE = 336                                       # 224 was 6 points worse alone, the same combined
GRID = SIZE // 14                                # patches per side


class Dino:
    def __init__(self, dev: str | None = None):
        import timm
        self.dev = dev or device()
        self.half = self.dev in ("mps", "cuda")
        model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE)
        cfg = timm.data.resolve_data_config({}, model=model)
        dtype = torch.float16 if self.half else torch.float32
        with GPU:
            model = model.to(self.dev).eval()
            self.model = model.half() if self.half else model
            self.mean = torch.tensor(cfg["mean"], dtype=dtype, device=self.dev).view(1, 3, 1, 1)
            self.std = torch.tensor(cfg["std"], dtype=dtype, device=self.dev).view(1, 3, 1, 1)
        self.dim = model.num_features

    @torch.no_grad()
    def images(self, ims: list[Image.Image]) -> np.ndarray:
        arrs = [np.asarray(on_white(i).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255 for i in ims]
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
        with GPU:
            x = x.to(self.dev)
            x = x.half() if self.half else x
            f = self.model((x - self.mean) / self.std).float()
            return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    @torch.no_grad()
    def patches(self, ims: list[Image.Image]) -> np.ndarray:
        """Each picture's patch vectors, unit length: (pictures, GRID * GRID, dim), row by row.
        Used to find the piece in a busy photo (SearchEngine._find_piece)."""
        arrs = [np.asarray(i.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255 for i in ims]
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
        with GPU:
            x = x.to(self.dev)
            x = x.half() if self.half else x
            t = self.model.forward_features((x - self.mean) / self.std)[:, self.model.num_prefix_tokens:].float()
            return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
