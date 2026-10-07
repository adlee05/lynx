from PIL import Image
import open_clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

print("Device:", device)
print("Model loaded")

# load image
image_path = "../dataset/val2017/000000106912.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_tensor)

print("Image embedding shape:", image_features.shape)

# query
query = "a person skateboarding on a sidewalk"
text_tokens = tokenizer([query]).to(device)

with torch.inference_mode():
    text_features = model.encode_text(text_tokens)

print("Text embedding shape:", text_features.shape)

# find similarity
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T).item()

print("Cosine similarity:", similarity)


unrelated_query = "a bowl of fruit on a table"
unrelated_tokens = tokenizer([unrelated_query]).to(device)

with torch.inference_mode():
    unrelated_features = model.encode_text(unrelated_tokens)

unrelated_features = (
    unrelated_features / unrelated_features.norm(dim=-1, keepdim=True)
)
unrelated_similarity = (image_features @ unrelated_features.T).item()

print("Unrelated query:", unrelated_query)
print("Unrelated similarity:", unrelated_similarity)
