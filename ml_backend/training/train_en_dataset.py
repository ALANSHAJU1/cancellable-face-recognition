import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.random_matrix import generate_random_matrix
from utils.celeba_dataloader import CelebADataset

# ======================================================
# CONFIG (FULL QUALITY MODE)
# ======================================================
torch.set_num_threads(os.cpu_count())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256
BATCH_SIZE = 8 if DEVICE == "cuda" else 4
EPOCHS = 100
LR = 1e-4
SUBSET_SIZE = 50000

FACE_DATASET_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba"
COVER_DATASET_PATH = "../datasets/covers"

CHECKPOINT_PATH = "../models/en_checkpoint.pth"
BEST_MODEL_PATH = "../models/extracting_network_dataset.pth"

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
    if f.lower().endswith((".jpg",".jpeg",".png"))
]

cover_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ======================================================
# MODELS
# ======================================================
# 🔒 Load trained Hiding Network (frozen)
hn = HidingNetwork().to(DEVICE)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth", map_location=DEVICE))
hn.eval()
for p in hn.parameters():
    p.requires_grad = False

# 🧠 Extracting Network (trainable)
en = ExtractingNetwork().to(DEVICE)

# 🎯 FaceNet supervision
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
for p in facenet.parameters():
    p.requires_grad = False

# 🔐 Random transformation matrix
R = generate_random_matrix().to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(en.parameters(), lr=LR)

# ======================================================
# RESUME SUPPORT (MID-EPOCH SAFE)
# ======================================================
start_epoch = 0
start_batch = 0
best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Resuming checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    en.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    start_batch = checkpoint.get("batch_idx", 0)
    best_val_loss = checkpoint["best_loss"]
    print(f"Resumed from epoch {start_epoch}, batch {start_batch}")

# ======================================================
# TRAIN LOOP
# ======================================================
train_losses = []
val_losses = []

for epoch in range(start_epoch, EPOCHS):

    en.train()
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

        # Stego generation (HN frozen)
        with torch.no_grad():
            stego = hn(faces, covers)

        # Extracting Network
        en_features = en(stego) @ R

        # FaceNet target
        faces_160 = torch.nn.functional.interpolate(faces, size=(160,160))
        target_features = facenet(faces_160)

        loss = criterion(en_features, target_features)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        # Mid-epoch checkpoint
        if batch_idx % 200 == 0:
            torch.save({
                "epoch": epoch,
                "batch_idx": batch_idx,
                "model_state": en.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_val_loss
            }, CHECKPOINT_PATH)

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ================= VALIDATION =================
    en.eval()
    val_loss = 0

    with torch.no_grad():
        for faces, _ in val_loader:
            faces = faces.to(DEVICE)

            dummy_cover = torch.zeros_like(faces)
            stego = hn(faces, dummy_cover)

            en_features = en(stego) @ R

            faces_160 = torch.nn.functional.interpolate(faces, size=(160,160))
            target_features = facenet(faces_160)

            val_loss += criterion(en_features, target_features).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(en.state_dict(), BEST_MODEL_PATH)
        print("💾 Saved best EN model")

    # End-epoch checkpoint
    torch.save({
        "epoch": epoch+1,
        "batch_idx": 0,
        "model_state": en.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_val_loss
    }, CHECKPOINT_PATH)

# ======================================================
# SAVE LOSS GRAPH
# ======================================================
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Extracting Network Training")
plt.savefig("../results/en_loss_graph.png")
plt.close()

print("........................ Extracting Network Training Completed Successfully!...............")

























# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torchvision import transforms
# from facenet_pytorch import InceptionResnetV1
# from PIL import Image
# import random

# from networks.hiding_network import HidingNetwork
# from networks.extracting_network import ExtractingNetwork
# from utils.random_matrix import generate_random_matrix
# from utils.celeba_dataloader import CelebADataset

# # ======================================================
# # CONFIGURATION (CPU / GPU SAFE)
# # ======================================================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BATCH_SIZE = 4
# EPOCHS = 2
# LR = 1e-4

# FACE_DATASET_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba"
# COVER_DATASET_PATH = "../datasets/covers"

# SUBSET_SIZE = 2000   # same subset size as HN training

# # ======================================================
# # DATASETS
# # ======================================================
# print("📂 Loading CelebA dataset...")

# face_dataset_full = CelebADataset(FACE_DATASET_PATH)
# face_dataset = Subset(face_dataset_full, range(SUBSET_SIZE))

# face_loader = DataLoader(
#     face_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# print(f"✅ Using {SUBSET_SIZE} face images for EN training")

# # Cover images
# cover_images = [
#     os.path.join(COVER_DATASET_PATH, f)
#     for f in os.listdir(COVER_DATASET_PATH)
#     if f.lower().endswith((".jpg", ".png"))
# ]

# if len(cover_images) == 0:
#     raise ValueError("No cover images found in datasets/covers")

# cover_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # ======================================================
# # MODELS
# # ======================================================
# print("🧠 Loading trained Hiding Network (frozen)...")
# hn = HidingNetwork().to(DEVICE)
# hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
# hn.eval()
# for p in hn.parameters():
#     p.requires_grad = False

# print("🧠 Initializing Extracting Network...")
# en = ExtractingNetwork().to(DEVICE)

# print("🧠 Loading FaceNet (feature target)...")
# facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
# for p in facenet.parameters():
#     p.requires_grad = False

# # Random matrix
# R = generate_random_matrix().to(DEVICE)

# # ======================================================
# # LOSS & OPTIMIZER
# # ======================================================
# criterion = nn.MSELoss()
# optimizer = optim.Adam(en.parameters(), lr=LR)

# # ======================================================
# # TRAINING LOOP
# # ======================================================
# print("🚀 Starting Extracting Network dataset training...\n")
# en.train()

# for epoch in range(EPOCHS):
#     epoch_loss = 0.0

#     for batch_idx, (faces, _) in enumerate(face_loader):
#         faces = faces.to(DEVICE)

#         # -------- Random cover batch --------
#         cover_batch = []
#         for _ in range(faces.size(0)):
#             cover_path = random.choice(cover_images)
#             cover_img = Image.open(cover_path).convert("RGB")
#             cover_img = cover_transform(cover_img)
#             cover_batch.append(cover_img)

#         covers = torch.stack(cover_batch).to(DEVICE)

#         # -------- Generate stego (HN frozen) --------
#         with torch.no_grad():
#             stego = hn(faces, covers)

#         # -------- Extract features --------
#         optimizer.zero_grad()

#         en_features = en(stego)
#         en_features = en_features @ R   # cancellable transform

#         # -------- FaceNet target --------
#         faces_160 = torch.nn.functional.interpolate(faces, size=(160, 160))
#         target_features = facenet(faces_160)

#         # -------- Loss --------
#         loss = criterion(en_features, target_features)

#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#         # -------- Progress logging --------
#         if batch_idx % 20 == 0:
#             print(
#                 f"Epoch {epoch+1}/{EPOCHS} | "
#                 f"Batch {batch_idx}/{len(face_loader)} | "
#                 f"Loss: {loss.item():.4f}"
#             )

#     avg_loss = epoch_loss / len(face_loader)
#     print(f"\n✅ Epoch [{epoch+1}/{EPOCHS}] completed — Avg Loss: {avg_loss:.4f}\n")

# # ======================================================
# # SAVE TRAINED EN MODEL
# # ======================================================
# os.makedirs("../models", exist_ok=True)
# torch.save(en.state_dict(), "../models/extracting_network_dataset.pth")

# print("💾 Dataset-trained Extracting Network saved at:")
# print("   ../models/extracting_network_dataset.pth")
# print("\n🎉 EN dataset training finished successfully")
