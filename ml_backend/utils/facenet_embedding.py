import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
FACE_IMAGE_PATH = "../datasets/processed_faces/processed_face_256.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# STEP 1: Load Pretrained FaceNet
# -------------------------------
facenet = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(DEVICE)

# -------------------------------
# STEP 2: Load Face Image
# -------------------------------
img = Image.open(FACE_IMAGE_PATH).convert("RGB")

# Resize to FaceNet expected size
img = img.resize((160, 160))

# Convert to tensor
img_tensor = torch.from_numpy(np.array(img)).float()

# Normalize to [-1, 1]
img_tensor = (img_tensor - 127.5) / 128.0

# Shape: (1, 3, 160, 160)
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# -------------------------------
# STEP 3: Extract Embedding
# -------------------------------
with torch.no_grad():
    embedding = facenet(img_tensor)

# -------------------------------
# STEP 4: Save Embedding
# -------------------------------
embedding_np = embedding.cpu().numpy()

output_path = "../datasets/processed_faces/facenet_embedding.npy"
np.save(output_path, embedding_np)

print("✅ FaceNet embedding extracted successfully")
print("📐 Embedding shape:", embedding_np.shape)
print("📁 Saved at:", output_path)
