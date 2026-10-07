"""Evaluate text-to-image retrieval on the COCO 2017 validation captions.

Uses the cached image embeddings produced by clip_smoke.py and compares exact
FAISS inner-product search with FAISS HNSW on the same normalized vectors.
"""

import json
import statistics
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from qdrant_client import QdrantClient, models


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_PATH = HERE / "image_embeddings.npz"
CAPTIONS_PATH = ROOT / "dataset" / "annotations" / "captions_val2017.json"
OUTPUT_PATH = HERE / "evaluation_results.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
TOP_KS = (1, 5, 10)
TEXT_BATCH_SIZE = 64
HNSW_M = 32
HNSW_EF_SEARCH = 64
QDRANT_COLLECTION = "lynx_eval_images"


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p))


def main() -> None:
    if not CACHE_PATH.exists():
        raise SystemExit(f"Missing image embedding cache: {CACHE_PATH}. Run clip_smoke.py first.")
    if not CAPTIONS_PATH.exists():
        raise SystemExit(
            f"Missing COCO captions: {CAPTIONS_PATH}. Add captions_val2017.json there first."
        )

    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        model_name = str(cache["model_name"].item())
        pretrained = str(cache["pretrained"].item())
        image_names = cache["image_names"].astype(str).tolist()
        image_vectors = np.asarray(cache["image_features"], dtype=np.float32)

    if model_name != MODEL_NAME or pretrained != PRETRAINED:
        raise SystemExit(
            f"Cache model is {model_name}/{pretrained}, expected {MODEL_NAME}/{PRETRAINED}."
        )

    with CAPTIONS_PATH.open(encoding="utf-8") as file:
        coco = json.load(file)

    # COCO filenames are the zero-padded image IDs, e.g. 000000106912.jpg.
    name_to_position = {name: pos for pos, name in enumerate(image_names)}
    captions: list[str] = []
    target_positions: list[int] = []
    for annotation in coco["annotations"]:
        image_name = f"{int(annotation['image_id']):012d}.jpg"
        position = name_to_position.get(image_name)
        if position is not None:
            captions.append(annotation["caption"].strip())
            target_positions.append(position)

    if not captions:
        raise SystemExit("No COCO captions matched the images in the embedding cache.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    text_batches: list[np.ndarray] = []
    embedding_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(captions), TEXT_BATCH_SIZE):
            batch = captions[start : start + TEXT_BATCH_SIZE]
            tokens = tokenizer(batch).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            text_batches.append(features.float().cpu().numpy())
    text_embedding_seconds = time.perf_counter() - embedding_started
    text_vectors = np.ascontiguousarray(np.concatenate(text_batches), dtype=np.float32)

    image_vectors = np.ascontiguousarray(image_vectors, dtype=np.float32)
    faiss.normalize_L2(image_vectors)

    exact = faiss.IndexFlatIP(image_vectors.shape[1])
    exact_build_started = time.perf_counter()
    exact.add(image_vectors)
    exact_build_seconds = time.perf_counter() - exact_build_started

    hnsw = faiss.IndexHNSWFlat(image_vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH
    hnsw_build_started = time.perf_counter()
    hnsw.add(image_vectors)
    hnsw_build_seconds = time.perf_counter() - hnsw_build_started

    max_k = min(max(TOP_KS), len(image_names))
    exact_positions = np.empty((len(captions), max_k), dtype=np.int64)
    hnsw_positions = np.empty((len(captions), max_k), dtype=np.int64)
    exact_latencies_ms: list[float] = []
    hnsw_latencies_ms: list[float] = []

    # Search one query at a time so the latency numbers resemble a single request.
    for query_number, vector in enumerate(text_vectors):
        query = vector.reshape(1, -1)

        started = time.perf_counter_ns()
        _, positions = exact.search(query, max_k)
        exact_latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        exact_positions[query_number] = positions[0]

        started = time.perf_counter_ns()
        _, positions = hnsw.search(query, max_k)
        hnsw_latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        hnsw_positions[query_number] = positions[0]

    # Use Qdrant's persistent local client mode in a temporary directory. This
    # measures Qdrant's local vector store without requiring a server or Docker.
    qdrant_positions = np.empty((len(captions), max_k), dtype=np.int64)
    qdrant_latencies_ms: list[float] = []
    image_position_by_name = {name: pos for pos, name in enumerate(image_names)}
    with tempfile.TemporaryDirectory(prefix="lynx-qdrant-eval-") as qdrant_path:
        qdrant = QdrantClient(path=qdrant_path)
        qdrant_setup_started = time.perf_counter()
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=image_vectors.shape[1],
                distance=models.Distance.COSINE,
            ),
        )
        qdrant_setup_seconds = time.perf_counter() - qdrant_setup_started

        qdrant_build_started = time.perf_counter()
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=position,
                    vector=vector.tolist(),
                    payload={"filename": image_names[position], "owner_id": "benchmark"},
                )
                for position, vector in enumerate(image_vectors)
            ],
            wait=True,
        )
        qdrant_upsert_seconds = time.perf_counter() - qdrant_build_started

        for query_number, vector in enumerate(text_vectors):
            started = time.perf_counter_ns()
            points = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector.tolist(),
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="owner_id",
                            match=models.MatchValue(value="benchmark"),
                        )
                    ]
                ),
                limit=max_k,
                with_payload=["filename"],
            ).points
            qdrant_latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            qdrant_positions[query_number] = [
                image_position_by_name[point.payload["filename"]]
                for point in points
            ]
        qdrant.close()

    targets = np.asarray(target_positions, dtype=np.int64)

    def quality_metrics(ranked_positions: np.ndarray) -> dict[str, float]:
        metrics: dict[str, float] = {}
        ranks = np.full(len(targets), np.inf)
        matches = ranked_positions == targets[:, None]
        rows, cols = np.where(matches)
        ranks[rows] = cols + 1
        for k in TOP_KS:
            effective_k = min(k, ranked_positions.shape[1])
            metrics[f"recall@{k}"] = float(np.mean(ranks <= effective_k))
        metrics["mrr@10"] = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
        return metrics

    ann_recall: dict[str, float] = {}
    qdrant_agreement: dict[str, float] = {}
    for k in TOP_KS:
        effective_k = min(k, max_k)
        intersections = [
            len(set(exact_positions[i, :effective_k]) & set(hnsw_positions[i, :effective_k]))
            / effective_k
            for i in range(len(captions))
        ]
        ann_recall[f"ann_recall@{k}"] = float(np.mean(intersections))
        qdrant_intersections = [
            len(set(exact_positions[i, :effective_k]) & set(qdrant_positions[i, :effective_k]))
            / effective_k
            for i in range(len(captions))
        ]
        qdrant_agreement[f"top_k_overlap@{k}"] = float(np.mean(qdrant_intersections))

    def latency_stats(values: list[float]) -> dict[str, float]:
        return {
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
        }

    results = {
        "dataset": "COCO 2017 validation captions, restricted to cached image subset",
        "model": f"{MODEL_NAME}/{PRETRAINED}",
        "device": device,
        "image_count": len(image_names),
        "caption_query_count": len(captions),
        "embedding_dimension": int(image_vectors.shape[1]),
        "similarity": "cosine (L2-normalized vectors + inner product)",
        "hnsw_parameters": {"m": HNSW_M, "ef_search": HNSW_EF_SEARCH},
        "text_embedding": {
            "total_seconds_batched": text_embedding_seconds,
            "mean_ms_per_caption_batched": text_embedding_seconds * 1000 / len(captions),
        },
        "exact_flat": {
            "index_build_seconds": exact_build_seconds,
            "search_latency": latency_stats(exact_latencies_ms),
            "semantic_retrieval": quality_metrics(exact_positions),
        },
        "hnsw": {
            "index_build_seconds": hnsw_build_seconds,
            "search_latency": latency_stats(hnsw_latencies_ms),
            "semantic_retrieval": quality_metrics(hnsw_positions),
            "agreement_with_exact": ann_recall,
        },
        "qdrant_local": {
            "mode": "persistent local client, temporary disk path; no Qdrant server",
            "collection_setup_seconds": qdrant_setup_seconds,
            "upsert_seconds": qdrant_upsert_seconds,
            "search_latency": latency_stats(qdrant_latencies_ms),
            "semantic_retrieval": quality_metrics(qdrant_positions),
            "agreement_with_exact": qdrant_agreement,
        },
        "notes": [
            "Semantic recall uses the caption's own COCO image as the relevant target.",
            "HNSW agreement is overlap with exact top-k, not semantic relevance.",
            "Qdrant measurements use the Python local client, not a Qdrant server deployment.",
            "Local mode applies the owner filter but does not use Qdrant server payload indexes.",
            "Caption annotations provide weak relevance labels; results are not human judgments.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
