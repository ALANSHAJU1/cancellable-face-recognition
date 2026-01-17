import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import cv2
from mtcnn import MTCNN
from sklearn.metrics import confusion_matrix

from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.random_matrix import generate_random_matrix

# ======================================================
# CONFIGURATION
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_FOLDER = "../datasets/celeba/img_align_celeba/img_align_celeba"
COVER_FOLDER = "../datasets/covers"
REFERENCE_FEATURE = "../datasets/processed_faces/reference_feature.npy"

NUM_TEST_IMAGES = 50
THRESHOLD = 0.18

# ======================================================
# LOAD MODELS
# ======================================================
print("🔹 Loading models...")

hn = HidingNetwork().to(DEVICE)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
hn.eval()

en = ExtractingNetwork().to(DEVICE)
en.load_state_dict(torch.load("../models/extracting_network_dataset.pth"))
en.eval()

detector = MTCNN()
R = generate_random_matrix().to(DEVICE)

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def cosine_similarity(a, b):
    a = a / torch.norm(a)
    b = b / torch.norm(b)
    return torch.dot(a, b).item()

def preprocess_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]["box"]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (256, 256))
    face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    return face.unsqueeze(0).to(DEVICE)

def load_cover(cover_path):
    cover = cv2.imread(cover_path)
    if cover is None:
        return None

    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover = cv2.resize(cover, (256, 256))
    cover = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
    return cover.unsqueeze(0).to(DEVICE)

# ======================================================
# LOAD DATA
# ======================================================
face_files = sorted(os.listdir(FACE_FOLDER))[:NUM_TEST_IMAGES]
reference = torch.tensor(np.load(REFERENCE_FEATURE)).to(DEVICE)

cover_files = [
    os.path.join(COVER_FOLDER, f)
    for f in os.listdir(COVER_FOLDER)
    if f.lower().endswith((".jpg", ".png"))
]

if len(cover_files) == 0:
    raise ValueError("No cover images found in covers folder")

# ======================================================
# EVALUATION LOOP (ALL COVER IMAGES)
# ======================================================
print("\n📊 MULTI-COVER AUTHENTICATION EVALUATION")
print("========================================")

for cover_path in cover_files:
    print(f"\n🖼 Cover image: {os.path.basename(cover_path)}")

    cover = load_cover(cover_path)
    if cover is None:
        print("❌ Could not load cover image, skipping")
        continue

    scores = []
    y_true = []
    y_pred = []

    for i, img_name in enumerate(face_files):
        img_path = os.path.join(FACE_FOLDER, img_name)
        face = preprocess_face(img_path)

        if face is None:
            continue

        with torch.no_grad():
            stego = hn(face, cover)
            feat = en(stego) @ R

        score = cosine_similarity(feat.squeeze(), reference.squeeze())
        scores.append(score)

        # Simulated labels
        is_genuine = 1 if i < NUM_TEST_IMAGES // 2 else 0
        prediction = 1 if score >= THRESHOLD else 0

        y_true.append(is_genuine)
        y_pred.append(prediction)

    scores = np.array(scores)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)

    print("📈 Similarity Stats")
    print(f"   Min : {scores.min():.4f}")
    print(f"   Max : {scores.max():.4f}")
    print(f"   Avg : {scores.mean():.4f}")

    print("📊 Metrics")
    print(f"   Accuracy : {accuracy:.3f}")
    print(f"   FAR      : {far:.3f}")
    print(f"   FRR      : {frr:.3f}")

print("\n✅ Multi-cover evaluation completed")
