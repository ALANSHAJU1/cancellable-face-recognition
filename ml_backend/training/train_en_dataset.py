import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import random

from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.random_matrix import generate_random_matrix
from utils.celeba_dataloader import CelebADataset

# ======================================================
# CONFIGURATION (CPU / GPU SAFE)
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-4

FACE_DATASET_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba"
COVER_DATASET_PATH = "../datasets/covers"

SUBSET_SIZE = 2000   # same subset size as HN training

# ======================================================
# DATASETS
# ======================================================
print("📂 Loading CelebA dataset...")

face_dataset_full = CelebADataset(FACE_DATASET_PATH)
face_dataset = Subset(face_dataset_full, range(SUBSET_SIZE))

face_loader = DataLoader(
    face_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"✅ Using {SUBSET_SIZE} face images for EN training")

# Cover images
cover_images = [
    os.path.join(COVER_DATASET_PATH, f)
    for f in os.listdir(COVER_DATASET_PATH)
    if f.lower().endswith((".jpg", ".png"))
]

if len(cover_images) == 0:
    raise ValueError("No cover images found in datasets/covers")

cover_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ======================================================
# MODELS
# ======================================================
print("🧠 Loading trained Hiding Network (frozen)...")
hn = HidingNetwork().to(DEVICE)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
hn.eval()
for p in hn.parameters():
    p.requires_grad = False

print("🧠 Initializing Extracting Network...")
en = ExtractingNetwork().to(DEVICE)

print("🧠 Loading FaceNet (feature target)...")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
for p in facenet.parameters():
    p.requires_grad = False

# Random matrix
R = generate_random_matrix().to(DEVICE)

# ======================================================
# LOSS & OPTIMIZER
# ======================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(en.parameters(), lr=LR)

# ======================================================
# TRAINING LOOP
# ======================================================
print("🚀 Starting Extracting Network dataset training...\n")
en.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for batch_idx, (faces, _) in enumerate(face_loader):
        faces = faces.to(DEVICE)

        # -------- Random cover batch --------
        cover_batch = []
        for _ in range(faces.size(0)):
            cover_path = random.choice(cover_images)
            cover_img = Image.open(cover_path).convert("RGB")
            cover_img = cover_transform(cover_img)
            cover_batch.append(cover_img)

        covers = torch.stack(cover_batch).to(DEVICE)

        # -------- Generate stego (HN frozen) --------
        with torch.no_grad():
            stego = hn(faces, covers)

        # -------- Extract features --------
        optimizer.zero_grad()

        en_features = en(stego)
        en_features = en_features @ R   # cancellable transform

        # -------- FaceNet target --------
        faces_160 = torch.nn.functional.interpolate(faces, size=(160, 160))
        target_features = facenet(faces_160)

        # -------- Loss --------
        loss = criterion(en_features, target_features)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # -------- Progress logging --------
        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(face_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(face_loader)
    print(f"\n✅ Epoch [{epoch+1}/{EPOCHS}] completed — Avg Loss: {avg_loss:.4f}\n")

# ======================================================
# SAVE TRAINED EN MODEL
# ======================================================
os.makedirs("../models", exist_ok=True)
torch.save(en.state_dict(), "../models/extracting_network_dataset.pth")

print("💾 Dataset-trained Extracting Network saved at:")
print("   ../models/extracting_network_dataset.pth")
print("\n🎉 EN dataset training finished successfully")
