import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from networks.hiding_network import HidingNetwork
from facenet_pytorch import InceptionResnetV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256

# Load trained HN
hn = HidingNetwork().to(DEVICE)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth", map_location=DEVICE))
hn.eval()

# Load FaceNet
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# Load one sample face & cover
face_path = "../datasets/celeba/img_align_celeba/img_align_celeba/000001.jpg"
cover_path = "../datasets/covers/cover3.jpg"   # change to any valid cover

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

face = transform(Image.open(face_path).convert("RGB")).unsqueeze(0).to(DEVICE)
cover = transform(Image.open(cover_path).convert("RGB")).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    stego = hn(face, cover)

# Compute PSNR
mse = torch.mean((stego - cover) ** 2)
psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

# Identity similarity
face_160 = torch.nn.functional.interpolate(face, size=(160,160))
stego_160 = torch.nn.functional.interpolate(stego, size=(160,160))

f1 = facenet(face_160)
f2 = facenet(stego_160)

cos_sim = torch.nn.functional.cosine_similarity(f1, f2).item()

print("\n===== HN Evaluation =====")
print(f"PSNR: {psnr.item():.2f} dB")
print(f"Identity Cosine Similarity: {cos_sim:.4f}")

# Show images
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(face.squeeze().permute(1,2,0).cpu())
plt.title("Face")

plt.subplot(1,3,2)
plt.imshow(cover.squeeze().permute(1,2,0).cpu())
plt.title("Cover")

plt.subplot(1,3,3)
plt.imshow(stego.squeeze().permute(1,2,0).cpu())
plt.title("Stego")

plt.show()