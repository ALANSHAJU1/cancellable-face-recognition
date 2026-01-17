import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from facenet_pytorch import InceptionResnetV1

from networks.hiding_network import HidingNetwork

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# MODELS
# -------------------------------
hn = HidingNetwork().to(device)

# Pretrained VGG-19 (for perceptual loss)
vgg = models.vgg19(pretrained=True).features[:16].eval().to(device)
for p in vgg.parameters():
    p.requires_grad = False

# Pretrained FaceNet
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
for p in facenet.parameters():
    p.requires_grad = False

# -------------------------------
# LOSSES
# -------------------------------
mse_loss = nn.MSELoss()

# -------------------------------
# OPTIMIZER
# -------------------------------
optimizer = optim.Adam(hn.parameters(), lr=1e-4)

# -------------------------------
# DUMMY INPUT (replace with dataloader later)
# -------------------------------
face = torch.rand(1, 3, 256, 256).to(device)
cover = torch.rand(1, 3, 256, 256).to(device)

# -------------------------------
# TRAINING LOOP (SINGLE STEP DEMO)
# -------------------------------
hn.train()
optimizer.zero_grad()

# Forward
stego = hn(face, cover)

# 1️⃣ Reconstruction loss
loss_rec = mse_loss(stego, cover)

# 2️⃣ Perceptual loss
loss_perc = mse_loss(vgg(stego), vgg(cover))

# 3️⃣ Feature loss
face_resized = torch.nn.functional.interpolate(face, size=(160, 160))
stego_resized = torch.nn.functional.interpolate(stego, size=(160, 160))

feat_face = facenet(face_resized)
feat_stego = facenet(stego_resized)

loss_feat = mse_loss(feat_face, feat_stego)

# Total loss
total_loss = loss_rec + 0.1 * loss_perc + loss_feat

# Backpropagation
total_loss.backward()
optimizer.step()

print("✅ Hiding Network training step completed")
print(f"Loss: {total_loss.item():.4f}")

torch.save(hn.state_dict(), "../models/hiding_network.pth")
print("💾 Hiding Network model saved")

