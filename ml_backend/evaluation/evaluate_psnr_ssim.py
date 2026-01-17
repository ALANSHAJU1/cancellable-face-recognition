import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import cv2
from mtcnn import MTCNN
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from networks.hiding_network import HidingNetwork

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_IMAGE = "../datasets/celeba/img_align_celeba/img_align_celeba/sample.jpg"
COVER_IMAGE = "../datasets/covers/cover(2).jpg"

SAVE_DIR = "../results/stego_quality"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# LOAD MODEL
# ======================================================
hn = HidingNetwork().to(DEVICE)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
hn.eval()

detector = MTCNN()

# ======================================================
# FACE PREPROCESSING
# ======================================================
def preprocess_face(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Face image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]["box"]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (256, 256))
    face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    return face.unsqueeze(0).to(DEVICE)

# ======================================================
# LOAD IMAGES
# ======================================================
face = preprocess_face(FACE_IMAGE)

cover = cv2.imread(COVER_IMAGE)
if cover is None:
    raise ValueError("Cover image not found")

cover_rgb = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
cover_resized = cv2.resize(cover_rgb, (256, 256))
cover_tensor = torch.tensor(cover_resized).permute(2, 0, 1).float() / 255.0
cover_tensor = cover_tensor.unsqueeze(0).to(DEVICE)

# ======================================================
# GENERATE STEGO
# ======================================================
with torch.no_grad():
    stego_tensor = hn(face, cover_tensor)

# ======================================================
# CONVERT TO NUMPY
# ======================================================
cover_np = cover_resized.astype(np.uint8)
stego_np = (
    stego_tensor.squeeze(0)
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)
stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

# ======================================================
# SAVE IMAGES
# ======================================================
cv2.imwrite(os.path.join(SAVE_DIR, "cover.png"), cv2.cvtColor(cover_np, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(SAVE_DIR, "stego.png"), cv2.cvtColor(stego_np, cv2.COLOR_RGB2BGR))

# ======================================================
# METRICS
# ======================================================
psnr_value = peak_signal_noise_ratio(cover_np, stego_np, data_range=255)
ssim_value = structural_similarity(
    cover_np, stego_np, channel_axis=2, data_range=255
)

# ======================================================
# OUTPUT
# ======================================================
print("\n📊 STEGO IMAGE QUALITY METRICS")
print("--------------------------------")
print(f"PSNR : {psnr_value:.2f} dB")
print(f"SSIM : {ssim_value:.4f}")
print("\n🖼 Images saved to:", SAVE_DIR)
