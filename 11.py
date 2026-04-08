import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 👉 这里换成你自己的图片路径
image_path = "test.jpg"

# 1️⃣ 加载模型
model_name = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

print("Model type:", type(model))

# 2️⃣ 加载图片
image = Image.open(image_path).convert("RGB")

# 3️⃣ 预处理
inputs = processor(images=image, return_tensors="pt").to(device)

# 4️⃣ 前向传播
with torch.no_grad():
    features = model.get_image_features(**inputs)

features = torch.nn.functional.normalize(features, p=2, dim=1)

print("Feature shape:", features.shape)
print("Feature sample:", features[0][:5])