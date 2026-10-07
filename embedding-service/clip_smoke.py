from PIL import Image
import open_clip
import torch
import random
from pathlib import Path
import numpy as np
import faiss

# cache params
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
IMAGE_LIMIT = 500
SAMPLE_SEED = 42
BATCH_SIZE = 32

CACHE_PATH = Path(__file__).resolve().parent / "image_embeddings.npz"

device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = Path(__file__).resolve().parents[1] / "dataset" / "val2017"
all_image_paths = sorted(image_dir.glob("*.jpg"))
image_paths = random.Random(SAMPLE_SEED).sample(
    all_image_paths,
    k=min(IMAGE_LIMIT, len(all_image_paths)),
)

print(f"Found {len(all_image_paths)} images; selected {len(image_paths)}")


expected_image_names = [path.name for path in image_paths]
cached_features = None

if CACHE_PATH.exists():
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as cache:
            cache_matches = (
                str(cache["model_name"].item()) == MODEL_NAME
                and str(cache["pretrained"].item()) == PRETRAINED
                and cache["image_names"].tolist() == expected_image_names
            )

            if cache_matches:
                cached_features = torch.from_numpy(
                    cache["image_features"].copy()
                )
                print("Compatible image embedding cache found.")
            else:
                print("Cache settings or image list changed; rebuilding it.")
    except (OSError, KeyError, ValueError) as error:
        print(f"Could not read cache; rebuilding it: {error}")

model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("Device:", device)
print("Model loaded")

if cached_features is not None:
    image_features = cached_features
    valid_image_paths = image_paths
    print("Loaded image embeddings from cache.")
else:
    image_batches = []
    valid_image_paths = []

    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch_images = []
        batch_valid_paths = []

        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    batch_images.append(preprocess(image.convert("RGB")))
                batch_valid_paths.append(path)
            except Exception as error:
                print(f"Skipping {path.name}: {error}")

        if not batch_images:
            continue

        image_tensor = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            batch_features = model.encode_image(image_tensor)

        batch_features = batch_features / batch_features.norm(
            dim=-1, keepdim=True
        )
        image_batches.append(batch_features.float().cpu())
        valid_image_paths.extend(batch_valid_paths)

    image_features = torch.cat(image_batches)

    np.savez_compressed(
        CACHE_PATH,
        image_features=image_features.numpy(),
        image_names=np.array([path.name for path in valid_image_paths]),
        model_name=np.array(MODEL_NAME),
        pretrained=np.array(PRETRAINED),
    )
    print(f"Saved image embeddings to {CACHE_PATH}")

print("Image embedding shape:", image_features.shape)

image_vectors = np.ascontiguousarray(image_features.numpy(), dtype=np.float32)
index = faiss.IndexFlatIP(image_vectors.shape[1])
index.add(image_vectors)
print(f"FAISS index contains {index.ntotal} image vectors.")

while True:
    query = input("\nQuery> ").strip()
    if query.lower() in {"quit", "exit"}:
        break
    if not query:
        continue

    text_tokens = tokenizer([query]).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    )
    query_vector = np.ascontiguousarray(
        text_features.cpu().numpy(),
        dtype=np.float32,
    )

    scores, positions = index.search(
        query_vector,
        min(5, index.ntotal),
    )

    print(f"Top results for: {query}")
    for rank, (score, position) in enumerate(
        zip(scores[0], positions[0]), 1
    ):
        print(
            f"{rank}. {valid_image_paths[position].name} "
            f"cosine={score:.3f}"
        )
