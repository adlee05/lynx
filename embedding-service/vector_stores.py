"""Vector-store adapters used by the CLIP retrieval engine."""

import os
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


DEMO_OWNER_ID = "demo"
QDRANT_COLLECTION = "lynx_images"
QDRANT_PATH = Path(__file__).resolve().parent / "qdrant_data"


class VectorStore(Protocol):
    backend_name: str

    @property
    def count(self) -> int: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...

    def close(self) -> None: ...


class FaissVectorStore:
    backend_name = "faiss_flat_ip"

    def __init__(self, vectors: np.ndarray, image_names: list[str]) -> None:
        self.image_names = image_names
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores, positions = self.index.search(query_vector, min(top_k, self.count))
        return [
            (self.image_names[int(position)], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]

    def close(self) -> None:
        pass


class QdrantVectorStore:
    """Qdrant client store; uses persistent local mode unless QDRANT_URL is set."""

    backend_name = "qdrant"

    def __init__(self, vectors: np.ndarray, image_names: list[str]) -> None:
        from qdrant_client import QdrantClient, models

        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY"),
            )
            self.persistent_local_mode = False
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(QDRANT_PATH))
            self.persistent_local_mode = True

        self.models = models
        self._count = len(image_names)

        if not self.client.collection_exists(QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=vectors.shape[1],
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.persistent_local_mode:
            self.client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="owner_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # Stable IDs keep benchmark setup repeatable across runs.
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=position,
                    vector=vector.tolist(),
                    payload={"filename": image_names[position], "owner_id": DEMO_OWNER_ID},
                )
                for position, vector in enumerate(vectors)
            ],
            wait=True,
        )

    @property
    def count(self) -> int:
        return self._count

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        result = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector[0].tolist(),
            query_filter=self.models.Filter(
                must=[
                    self.models.FieldCondition(
                        key="owner_id",
                        match=self.models.MatchValue(value=DEMO_OWNER_ID),
                    )
                ]
            ),
            limit=min(top_k, self.count),
            with_payload=["filename"],
        ).points
        return [
            (str(point.payload["filename"]), float(point.score))
            for point in result
            if point.payload and "filename" in point.payload
        ]

    def close(self) -> None:
        self.client.close()


def create_vector_store(
    backend: str,
    vectors: np.ndarray,
    image_names: list[str],
) -> VectorStore:
    if backend == "faiss":
        return FaissVectorStore(vectors, image_names)
    if backend == "qdrant":
        return QdrantVectorStore(vectors, image_names)
    raise ValueError("LYNX_VECTOR_BACKEND must be 'faiss' or 'qdrant'")
