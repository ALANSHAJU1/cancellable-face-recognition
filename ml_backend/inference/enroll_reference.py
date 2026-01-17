import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
import cv2

from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.random_matrix import generate_random_matrix

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# LOAD MODELS
# -------------------------------
hn = HidingNetwork().to(device)
hn.load_state_dict(torch.load("../models/hiding_network.pth"))
hn.eval()

en = ExtractingNetwork().to(device)
en.load_state_dict(torch.load("../models/extracting_network.pth"))
en.eval()

detector = MTCNN()
R = generate_random_matrix().to(device)

# -------------------------------
# PREPROCESS FACE
# -------------------------------
def preprocess_face(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Face image not found at {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]["box"]
    face = img_rgb[y:y + h, x:x + w]

    face = cv2.resize(face, (256, 256))
    face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    return face.unsqueeze(0).to(device)

# -------------------------------
# ENROLLMENT FUNCTION
# -------------------------------
def enroll(face_image_path, cover_image_path, save_path):

    # Load face
    face = preprocess_face(face_image_path)

    # Load cover
    cover = cv2.imread(cover_image_path)
    if cover is None:
        raise ValueError(f"Cover image not found at {cover_image_path}")

    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover = cv2.resize(cover, (256, 256))
    cover = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
    cover = cover.unsqueeze(0).to(device)

    # Generate stego
    with torch.no_grad():
        stego = hn(face, cover)

    # Extract protected feature
    with torch.no_grad():
        feature = en(stego)
        feature = feature @ R

    # Save reference feature
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, feature.cpu().numpy())

    print("✅ Reference feature enrolled and saved at:", save_path)

# -------------------------------
# RUN ENROLLMENT
# -------------------------------
if __name__ == "__main__":
    enroll(
        face_image_path="../datasets/celeba/img_align_celeba/img_align_celeba/sample.jpg",
        cover_image_path="../datasets/covers/cover.jpg",
        save_path="../datasets/processed_faces/reference_feature.npy"
    )
