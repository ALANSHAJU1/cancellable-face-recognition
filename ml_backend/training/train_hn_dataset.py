import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import random

from networks.hiding_network import HidingNetwork
from utils.celeba_dataloader import CelebADataset

# ======================================================
# CONFIGURATION (SAFE FOR CPU & IEEE PROJECT)
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4        # small batch for CPU safety
EPOCHS = 2            # demonstration epochs (increase if GPU)
LR = 1e-4

FACE_DATASET_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba"
COVER_DATASET_PATH = "../datasets/covers"

SUBSET_SIZE = 2000    # <<< IMPORTANT: subset size for training speed

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

print(f"✅ Using {SUBSET_SIZE} face images for training")

# Load cover images list
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
print("🧠 Loading Hiding Network...")
hn = HidingNetwork().to(DEVICE)

print("🧠 Loading VGG-19 for perceptual loss...")
vgg = models.vgg19(pretrained=True).features[:16].eval().to(DEVICE)
for p in vgg.parameters():
    p.requires_grad = False

print("🧠 Loading FaceNet for feature loss...")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
for p in facenet.parameters():
    p.requires_grad = False

# ======================================================
# LOSS & OPTIMIZER
# ======================================================
mse_loss = nn.MSELoss()
optimizer = optim.Adam(hn.parameters(), lr=LR)

# ======================================================
# TRAINING LOOP
# ======================================================
print("🚀 Starting Hiding Network dataset training...\n")
hn.train()

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

        # -------- Forward + Backprop --------
        optimizer.zero_grad()

        stego = hn(faces, covers)

        # 1️⃣ Reconstruction loss
        loss_rec = mse_loss(stego, covers)

        # 2️⃣ Perceptual loss
        loss_perc = mse_loss(vgg(stego), vgg(covers))

        # 3️⃣ Feature loss
        faces_160 = torch.nn.functional.interpolate(faces, size=(160, 160))
        stego_160 = torch.nn.functional.interpolate(stego, size=(160, 160))

        feat_face = facenet(faces_160)
        feat_stego = facenet(stego_160)

        loss_feat = mse_loss(feat_face, feat_stego)

        # Total loss
        total_loss = loss_rec + 0.1 * loss_perc + loss_feat

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        # -------- Progress logging --------
        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(face_loader)} | "
                f"Loss: {total_loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(face_loader)
    print(f"\n✅ Epoch [{epoch+1}/{EPOCHS}] completed — Avg Loss: {avg_loss:.4f}\n")

# ======================================================
# SAVE TRAINED MODEL
# ======================================================
os.makedirs("../models", exist_ok=True)
torch.save(hn.state_dict(), "../models/hiding_network_dataset.pth")

print("💾 Dataset-trained Hiding Network saved at:")
print("   ../models/hiding_network_dataset.pth")
print("\n🎉 HN dataset training finished successfully")

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import models, transforms
# from facenet_pytorch import InceptionResnetV1
# from PIL import Image
# import random

# from networks.hiding_network import HidingNetwork
# from utils.celeba_dataloader import CelebADataset

# # -------------------------------
# # CONFIG
# # -------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 4
# EPOCHS = 5
# LR = 1e-4

# FACE_DATASET_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba"
# COVER_DATASET_PATH = "../datasets/covers"

# # -------------------------------
# # DATASETS
# # -------------------------------
# face_dataset = CelebADataset(FACE_DATASET_PATH)
# face_loader = DataLoader(face_dataset, batch_size=BATCH_SIZE, shuffle=True)

# cover_images = [
#     os.path.join(COVER_DATASET_PATH, f)
#     for f in os.listdir(COVER_DATASET_PATH)
#     if f.lower().endswith((".jpg", ".png"))
# ]

# cover_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # -------------------------------
# # MODELS
# # -------------------------------
# hn = HidingNetwork().to(DEVICE)

# vgg = models.vgg19(pretrained=True).features[:16].eval().to(DEVICE)
# for p in vgg.parameters():
#     p.requires_grad = False

# facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
# for p in facenet.parameters():
#     p.requires_grad = False

# # -------------------------------
# # LOSSES & OPTIMIZER
# # -------------------------------
# mse_loss = nn.MSELoss()
# optimizer = optim.Adam(hn.parameters(), lr=LR)

# # -------------------------------
# # TRAINING LOOP
# # -------------------------------
# hn.train()

# for epoch in range(EPOCHS):
#     epoch_loss = 0.0

#     for faces, _ in face_loader:
#         faces = faces.to(DEVICE)

#         # Random cover batch
#         cover_batch = []
#         for _ in range(faces.size(0)):
#             cover_path = random.choice(cover_images)
#             cover_img = Image.open(cover_path).convert("RGB")
#             cover_img = cover_transform(cover_img)
#             cover_batch.append(cover_img)

#         covers = torch.stack(cover_batch).to(DEVICE)

#         optimizer.zero_grad()

#         # Forward
#         stego = hn(faces, covers)

#         # 1️⃣ Reconstruction loss
#         loss_rec = mse_loss(stego, covers)

#         # 2️⃣ Perceptual loss
#         loss_perc = mse_loss(vgg(stego), vgg(covers))

#         # 3️⃣ Feature loss
#         faces_160 = torch.nn.functional.interpolate(faces, size=(160, 160))
#         stego_160 = torch.nn.functional.interpolate(stego, size=(160, 160))

#         feat_face = facenet(faces_160)
#         feat_stego = facenet(stego_160)

#         loss_feat = mse_loss(feat_face, feat_stego)

#         # Total loss
#         total_loss = loss_rec + 0.1 * loss_perc + loss_feat

#         total_loss.backward()
#         optimizer.step()

#         epoch_loss += total_loss.item()

#     avg_loss = epoch_loss / len(face_loader)
#     print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# # -------------------------------
# # SAVE MODEL
# # -------------------------------
# os.makedirs("../models", exist_ok=True)
# torch.save(hn.state_dict(), "../models/hiding_network_dataset.pth")
# print("💾 Dataset-trained Hiding Network saved")
