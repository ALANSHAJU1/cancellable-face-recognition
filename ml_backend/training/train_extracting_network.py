import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1

from networks.extracting_network import ExtractingNetwork
from networks.hiding_network import HidingNetwork
from utils.random_matrix import generate_random_matrix

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# MODELS
# -------------------------------
hn = HidingNetwork().to(device)
hn.load_state_dict(torch.load("../models/hiding_network.pth"))
hn.eval()
for p in hn.parameters():
    p.requires_grad = False

en = ExtractingNetwork().to(device)

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
for p in facenet.parameters():
    p.requires_grad = False

# -------------------------------
# RANDOM MATRIX
# -------------------------------
R = generate_random_matrix().to(device)

# -------------------------------
# OPTIMIZER & LOSS
# -------------------------------
optimizer = optim.Adam(en.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# -------------------------------
# DUMMY INPUT (replace with dataloader later)
# -------------------------------
face = torch.rand(1, 3, 256, 256).to(device)
cover = torch.rand(1, 3, 256, 256).to(device)

# -------------------------------
# TRAINING STEP
# -------------------------------
optimizer.zero_grad()

with torch.no_grad():
    stego = hn(face, cover)

# Extract features
en_features = en(stego)

# Random matrix transform
en_features_protected = en_features @ R

# FaceNet target features
face_resized = torch.nn.functional.interpolate(face, size=(160, 160))
target_features = facenet(face_resized)

# Loss
loss = criterion(en_features_protected, target_features)

loss.backward()
optimizer.step()

print("✅ Extracting Network training step completed")
print(f"Loss: {loss.item():.4f}")

# -------------------------------
# SAVE TRAINED EXTRACTING NETWORK
# -------------------------------
os.makedirs("../models", exist_ok=True)
torch.save(en.state_dict(), "../models/extracting_network.pth")
print("💾 Extracting Network model saved")

