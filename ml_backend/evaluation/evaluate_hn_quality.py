# evaluate_hn_quality.py
import sys
import os

# Fix import path so we can import networks/ and utils/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from networks.hiding_network import HidingNetwork
from utils.celeba_dataloader import CelebADataset

# CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
NUM_SAMPLES = 200  # number of samples for evaluation

FACE_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "celeba",
    "img_align_celeba",
    "img_align_celeba"
)

COVER_PATH = os.path.join(BASE_DIR, "datasets", "covers")

MODEL_A_PATH = os.path.join(BASE_DIR, "models", "hiding_network.pth")
MODEL_B_PATH = os.path.join(BASE_DIR, "models", "hiding_network_dataset.pth")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model = HidingNetwork().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------
# RANDOM COVER IMAGE
# ---------------------------------------------------------
def get_random_cover():
    cover_files = [
        f for f in os.listdir(COVER_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(cover_files) == 0:
        raise ValueError("No cover images found in covers directory.")

    img_path = os.path.join(COVER_PATH, np.random.choice(cover_files))
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate(model_path):
    print("\n" + "=" * 60)
    print(f"Evaluating model: {os.path.basename(model_path)}")
    print("=" * 60)

    model = load_model(model_path)
    dataset = CelebADataset(FACE_PATH)

    psnr_vals = []
    ssim_vals = []

    with torch.no_grad():
        for i in tqdm(range(min(NUM_SAMPLES, len(dataset)))):
            face, _ = dataset[i]
            face = face.unsqueeze(0).to(DEVICE)
            cover = get_random_cover()

            stego = model(face, cover)

            cover_np = cover.squeeze().cpu().permute(1, 2, 0).numpy()
            stego_np = stego.squeeze().cpu().permute(1, 2, 0).numpy()

            psnr_vals.append(
                peak_signal_noise_ratio(cover_np, stego_np, data_range=1.0)
            )

            ssim_vals.append(
                structural_similarity(
                    cover_np,
                    stego_np,
                    channel_axis=2,
                    data_range=1.0
                )
            )

    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)

    print(f"\nAverage PSNR : {avg_psnr:.4f} dB")
    print(f"Average SSIM : {avg_ssim:.4f}")

    return avg_psnr, avg_ssim


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    print("\nStarting Hiding Network Quality Comparison...")

    psnr_a, ssim_a = evaluate(MODEL_A_PATH)
    psnr_b, ssim_b = evaluate(MODEL_B_PATH)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(f"Model A (hiding_network.pth)")
    print(f"   PSNR: {psnr_a:.4f} dB")
    print(f"   SSIM: {ssim_a:.4f}")

    print("\nModel B (hiding_network_dataset.pth)")
    print(f"   PSNR: {psnr_b:.4f} dB")
    print(f"   SSIM: {ssim_b:.4f}")

    print("\nEvaluation Complete.")