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

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
detector = MTCNN()

R = generate_random_matrix().to(device)

# -------------------------------
# SIMILARITY FUNCTION
# -------------------------------
def cosine_similarity(a, b):
    a = a / torch.norm(a)
    b = b / torch.norm(b)
    return torch.dot(a, b).item()

# -------------------------------
# PREPROCESS FACE (MTCNN)
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
# AUTHENTICATION FUNCTION
# -------------------------------
def authenticate(live_face_path, cover_image_path, reference_feature_path, threshold=0.7):

    # ---- Load & preprocess face ----
    face = preprocess_face(live_face_path)

    # ---- Load cover image ----
    cover = cv2.imread(cover_image_path)
    if cover is None:
        raise ValueError(f"Cover image not found at {cover_image_path}")

    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover = cv2.resize(cover, (256, 256))
    cover = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
    cover = cover.unsqueeze(0).to(device)

    # ---- Generate stego image ----
    with torch.no_grad():
        stego = hn(face, cover)

    # ---- Extract protected feature ----
    with torch.no_grad():
        feat = en(stego)
        feat = feat @ R   # random matrix transformation

    # ---- Load reference feature ----
    ref = torch.tensor(np.load(reference_feature_path)).to(device)

    # ---- Similarity computation ----
    score = cosine_similarity(feat.squeeze(), ref.squeeze())
    decision = "ACCEPT" if score >= threshold else "REJECT"

    return score, decision

# -------------------------------
# TEST RUN
# -------------------------------
if __name__ == "__main__":
    score, decision = authenticate(
        live_face_path="../datasets/celeba/img_align_celeba/img_align_celeba/sample.jpg",
        cover_image_path=os.path.abspath("../datasets/covers/cover.png"),
        reference_feature_path="../datasets/processed_faces/reference_feature.npy"
    )

    print("Similarity score:", score)
    print("Decision:", decision)
