from PIL import Image
import open_clip
import torch
import random
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = Path(__file__).resolve().parents[1] / "dataset" / "val2017"
all_image_paths = sorted(image_dir.glob("*.jpg"))
image_paths = random.Random(42).sample(
    all_image_paths,
    k=min(500, len(all_image_paths)),
)
BATCH_SIZE = 32

print(f"Found {len(all_image_paths)} images; selected {len(image_paths)}")

model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

print("Device:", device)
print("Model loaded")

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
        image_features = model.encode_image(image_tensor)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_batches.append(image_features.float().cpu())
    valid_image_paths.extend(batch_valid_paths)

image_features = torch.cat(image_batches)
print("Image embedding shape:", image_features.shape)

# query
query = "a cat"
text_tokens = tokenizer([query]).to(device)

with torch.inference_mode():
    text_features = model.encode_text(text_tokens)

print("Text embedding shape:", text_features.shape)

# find similarity
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarities = (image_features @ text_features.cpu().T).squeeze(1)
top_scores, top_positions = similarities.topk(k=min(5, len(valid_image_paths)))

print(f"Top results for: {query}")
for rank, (score, position) in enumerate(zip(top_scores.tolist(), top_positions.tolist()), 1):
    print(f"{rank}. {valid_image_paths[position].name}  cosine={score:.3f}")
