#ml_backend/training/train_hn_dataset.py
import sys, os, random, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.hiding_network import HidingNetwork
from utils.celeba_dataloader import CelebADataset

# ======================================================
# SYSTEM CONFIG
# ======================================================
torch.set_num_threads(os.cpu_count())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256
BATCH_SIZE = 8 if DEVICE == "cuda" else 4
EPOCHS = 100
LR = 1e-4
SUBSET_SIZE = 20000

FACE_DATASET_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba"
COVER_DATASET_PATH = "../datasets/covers"

CHECKPOINT_PATH = "../models/hn_checkpoint.pth"
BEST_MODEL_PATH = "../models/hiding_network_dataset.pth"

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ======================================================
# DATASET
# ======================================================
dataset_full = CelebADataset(FACE_DATASET_PATH)
dataset = Subset(dataset_full, range(SUBSET_SIZE))

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 if DEVICE=="cuda" else 0,
    pin_memory=(DEVICE=="cuda")
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ======================================================
# SAFE COVER LIST
# ======================================================
cover_files = [
    os.path.join(COVER_DATASET_PATH, f)
    for f in os.listdir(COVER_DATASET_PATH)
    if os.path.isfile(os.path.join(COVER_DATASET_PATH, f))
    and f.lower().endswith((".jpg",".jpeg",".png"))
]

if len(cover_files) == 0:
    raise ValueError("No valid cover images found.")

cover_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ======================================================
# MODELS
# ======================================================
hn = HidingNetwork().to(DEVICE)

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16].eval().to(DEVICE)
for p in vgg.parameters():
    p.requires_grad = False

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
for p in facenet.parameters():
    p.requires_grad = False

criterion = nn.MSELoss()
optimizer = optim.Adam(hn.parameters(), lr=LR)

# ======================================================
# RESUME SUPPORT
# ======================================================
start_epoch = 0
start_batch = 0
best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Resuming checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    hn.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    start_batch = checkpoint.get("batch_idx", 0)
    best_val_loss = checkpoint["best_loss"]
    print(f"Resumed from epoch {start_epoch}, batch {start_batch}")

# ======================================================
# TRAIN LOOP WITH SAFE INTERRUPT SAVE
# ======================================================
train_losses, val_losses = [], []

try:
    for epoch in range(start_epoch, EPOCHS):

        hn.train()
        running_loss = 0

        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (faces, _) in pbar:

            if epoch == start_epoch and batch_idx < start_batch:
                continue

            faces = faces.to(DEVICE)

            # Random cover batch
            cover_batch = []
            for _ in range(faces.size(0)):
                img = Image.open(random.choice(cover_files)).convert("RGB")
                img = cover_transform(img)
                cover_batch.append(img)

            covers = torch.stack(cover_batch).to(DEVICE)

            optimizer.zero_grad()

            stego = hn(faces, covers)

            loss_rec = criterion(stego, covers)
            loss_perc = criterion(vgg(stego), vgg(covers))

            faces_160 = torch.nn.functional.interpolate(faces, size=(160,160))
            stego_160 = torch.nn.functional.interpolate(stego, size=(160,160))

            loss_feat = criterion(
                facenet(faces_160),
                facenet(stego_160)
            )

            total_loss = loss_rec + 0.1*loss_perc + loss_feat

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

            # Mid-epoch checkpoint every 200 batches
            if batch_idx % 200 == 0:
                torch.save({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "model_state": hn.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_val_loss
                }, CHECKPOINT_PATH)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ================= VALIDATION =================
        hn.eval()
        val_loss = 0

        with torch.no_grad():
            for faces, _ in val_loader:
                faces = faces.to(DEVICE)
                dummy_cover = torch.zeros_like(faces)
                stego = hn(faces, dummy_cover)
                val_loss += criterion(stego, dummy_cover).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(hn.state_dict(), BEST_MODEL_PATH)
            print("💾 Saved best model")

        # End-epoch checkpoint
        torch.save({
            "epoch": epoch+1,
            "batch_idx": 0,
            "model_state": hn.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_val_loss
        }, CHECKPOINT_PATH)

except KeyboardInterrupt:
    print("\n⚠ Training interrupted! Saving checkpoint safely...")
    torch.save({
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state": hn.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_val_loss
    }, CHECKPOINT_PATH)
    print("✅ Safe checkpoint saved.")
    exit()

# ======================================================
# SAVE LOSS GRAPH
# ======================================================
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.savefig("../results/hn_loss_graph.png")
plt.close()

print("....................................... HN Training Complete...............................")













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
