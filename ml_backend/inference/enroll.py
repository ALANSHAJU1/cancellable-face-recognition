import sys
import cv2
import numpy as np
from utils.encrypt_and_store import encrypt_and_store_stego
from networks.hiding_network import HidingNetwork
import torch

user_id = sys.argv[1]

# Load HN
device = "cuda" if torch.cuda.is_available() else "cpu"
hn = HidingNetwork().to(device)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
hn.eval()

# Dummy face + cover (replace with real capture if needed)
face = torch.rand(1, 3, 256, 256).to(device)
cover = torch.rand(1, 3, 256, 256).to(device)

# Generate stego
with torch.no_grad():
    stego = hn(face, cover)

# Convert stego to PNG bytes
stego_np = stego.squeeze().permute(1, 2, 0).cpu().numpy()
stego_np = (stego_np * 255).astype(np.uint8)
_, stego_png = cv2.imencode(".png", stego_np)

# Dummy quality metrics (replace with real PSNR/SSIM if needed)
psnr = 38.2
ssim = 0.96
cover_hash = "dummy_hash"

# Encrypt & store
encrypt_and_store_stego(
    user_id=user_id,
    stego_png_bytes=stego_png.tobytes(),
    psnr=psnr,
    ssim=ssim,
    cover_hash=cover_hash
)

print("Enrollment completed for user:", user_id)
