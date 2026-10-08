"""SigLIP2 image/text encoder shared by indexing and search."""
import numpy as np
import open_clip
import torch
from PIL import Image

from .config import GPU, MODEL_NAME, MODEL_PRETRAINED, device


def on_white(im: Image.Image) -> Image.Image:
    """Renders have transparent backgrounds; flatten onto white for the model."""
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


class Embedder:
    def __init__(self, dev: str | None = None):
        self.dev = dev or device()
        with GPU:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=MODEL_PRETRAINED, device=self.dev)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    @torch.no_grad()
    def images(self, ims: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.preprocess(on_white(i)) for i in ims])
        with GPU:
            f = self.model.encode_image(x.to(self.dev))
            return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()

    @torch.no_grad()
    def texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts)
        with GPU:
            f = self.model.encode_text(tokens.to(self.dev))
            return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()
