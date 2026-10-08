"""CLIP image and text embedding functions. Search and accounts live in Go."""

from io import BytesIO

import numpy as np
import open_clip
import torch
from PIL import Image


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class EmbeddingEngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    @property
    def dimension(self) -> int:
        return int(self.model.visual.output_dim)

    def embed_text(self, text: str) -> list[float]:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
        return self._normalize(features)

    def embed_image(self, image_bytes: bytes) -> list[float]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensor)
        return self._normalize(features)

    @staticmethod
    def _normalize(features: torch.Tensor) -> list[float]:
        features = features / features.norm(dim=-1, keepdim=True)
        return np.asarray(features.float().cpu().numpy()[0], dtype=np.float32).tolist()
