"""Reusable CLIP + FAISS retrieval engine for the Lynx API."""

from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVICE_DIR.parent
CACHE_PATH = SERVICE_DIR / "image_embeddings.npz"
IMAGE_DIR = PROJECT_DIR / "dataset" / "val2017"


class RetrievalEngine:
    """Loads CLIP once and serves searches against cached image embeddings."""

    def __init__(self) -> None:
        if not CACHE_PATH.exists():
            raise FileNotFoundError(
                f"Image embedding cache not found: {CACHE_PATH}. "
                "Run clip_smoke.py once to generate it."
            )

        with np.load(CACHE_PATH, allow_pickle=False) as cache:
            cached_model = str(cache["model_name"].item())
            cached_pretrained = str(cache["pretrained"].item())
            self.image_names = cache["image_names"].astype(str).tolist()
            image_vectors = np.asarray(cache["image_features"], dtype=np.float32)

        if cached_model != MODEL_NAME or cached_pretrained != PRETRAINED:
            raise ValueError(
                "Embedding cache model does not match the API model: "
                f"{cached_model}/{cached_pretrained}"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        # Stored image vectors and incoming text vectors are both normalized,
        # so inner-product ranking is equivalent to cosine-similarity ranking.
        self.image_vectors = np.ascontiguousarray(image_vectors, dtype=np.float32)
        faiss.normalize_L2(self.image_vectors)
        self.index = faiss.IndexFlatIP(self.image_vectors.shape[1])
        self.index.add(self.image_vectors)

        self.image_paths = {
            name: IMAGE_DIR / name
            for name in self.image_names
            if (IMAGE_DIR / name).is_file()
        }

    @property
    def image_count(self) -> int:
        return self.index.ntotal

    @property
    def model_name(self) -> str:
        return f"{MODEL_NAME}/{PRETRAINED}"

    @property
    def embedding_dimension(self) -> int:
        return self.image_vectors.shape[1]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, float | int | str]]:
        tokens = self.tokenizer([query]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        query_vector = np.ascontiguousarray(
            text_features.float().cpu().numpy(), dtype=np.float32
        )
        scores, positions = self.index.search(query_vector, min(top_k, self.image_count))

        results: list[dict[str, float | int | str]] = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position < 0:
                continue
            filename = self.image_names[int(position)]
            results.append(
                {
                    "rank": rank,
                    "filename": filename,
                    "image_url": f"/images/{filename}",
                    "score": float(score),
                }
            )
        return results
