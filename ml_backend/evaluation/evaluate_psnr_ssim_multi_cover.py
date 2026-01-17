import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import cv2
from mtcnn import MTCNN
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from networks.hiding_network import HidingNetwork

# ======================================================
# CONFIGURATION
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_IMAGE = "../datasets/celeba/img_align_celeba/img_align_celeba/sample.jpg"
COVER_FOLDER = "../datasets/covers"

SAVE_DIR = "../results/stego_quality_multi_cover"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# LOAD MODEL
# ======================================================
print("🔹 Loading Hiding Network...")
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
        raise ValueError(f"Face image not found at {path}")

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
# LOAD FACE ONCE
# ======================================================
face = preprocess_face(FACE_IMAGE)

# ======================================================
# LOAD ALL COVER IMAGES
# ======================================================
cover_files = [
    os.path.join(COVER_FOLDER, f)
    for f in os.listdir(COVER_FOLDER)
    if f.lower().endswith((".jpg", ".png"))
]

if len(cover_files) == 0:
    raise ValueError("No cover images found in covers folder")

# ======================================================
# MULTI-COVER PSNR / SSIM EVALUATION
# ======================================================
print("\n📊 PSNR & SSIM RESULTS (MULTI-COVER)")
print("===================================")

for cover_path in cover_files:
    cover_name = os.path.splitext(os.path.basename(cover_path))[0]

    cover = cv2.imread(cover_path)
    if cover is None:
        print(f"❌ Could not load {cover_path}, skipping")
        continue

    cover_rgb = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover_resized = cv2.resize(cover_rgb, (256, 256))
    cover_tensor = torch.tensor(cover_resized).permute(2, 0, 1).float() / 255.0
    cover_tensor = cover_tensor.unsqueeze(0).to(DEVICE)

    # Generate stego
    with torch.no_grad():
        stego_tensor = hn(face, cover_tensor)

    # Convert to numpy
    stego_np = (
        stego_tensor.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)
    cover_np = cover_resized.astype(np.uint8)

    # Save images
    cover_save = os.path.join(SAVE_DIR, f"{cover_name}_cover.png")
    stego_save = os.path.join(SAVE_DIR, f"{cover_name}_stego.png")

    cv2.imwrite(cover_save, cv2.cvtColor(cover_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(stego_save, cv2.cvtColor(stego_np, cv2.COLOR_RGB2BGR))

    # Metrics
    psnr_value = peak_signal_noise_ratio(cover_np, stego_np, data_range=255)
    ssim_value = structural_similarity(
        cover_np, stego_np, channel_axis=2, data_range=255
    )

    print(f"\n🖼 Cover: {os.path.basename(cover_path)}")
    print(f"   PSNR : {psnr_value:.2f} dB")
    print(f"   SSIM : {ssim_value:.4f}")

print("\n✅ Multi-cover PSNR/SSIM evaluation completed")
print(f"🖼 Images saved in: {SAVE_DIR}")
