import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from mtcnn import MTCNN
import cv2
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import confusion_matrix

from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.random_matrix import generate_random_matrix

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.18
NUM_TEST_IMAGES = 50   # number of images to evaluate

FACE_FOLDER = "../datasets/celeba/img_align_celeba/img_align_celeba"
COVER_IMAGE = "../datasets/covers/cover.jpg"
REFERENCE_FEATURE = "../datasets/processed_faces/reference_feature.npy"

# ======================================================
# LOAD MODELS
# ======================================================
hn = HidingNetwork().to(DEVICE)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
hn.eval()

en = ExtractingNetwork().to(DEVICE)
en.load_state_dict(torch.load("../models/extracting_network_dataset.pth"))
en.eval()

detector = MTCNN()
R = generate_random_matrix().to(DEVICE)

# ======================================================
# HELPERS
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

def load_cover():
    cover = cv2.imread(COVER_IMAGE)
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover = cv2.resize(cover, (256, 256))
    cover = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
    return cover.unsqueeze(0).to(DEVICE)

# ======================================================
# EVALUATION
# ======================================================
face_files = sorted(os.listdir(FACE_FOLDER))[:NUM_TEST_IMAGES]
reference = torch.tensor(np.load(REFERENCE_FEATURE)).to(DEVICE)

y_true = []
y_pred = []
scores = []

cover = load_cover()

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

    # First half = genuine, second half = impostor (simulation)
    is_genuine = 1 if i < NUM_TEST_IMAGES // 2 else 0
    prediction = 1 if score >= THRESHOLD else 0

    y_true.append(is_genuine)
    y_pred.append(prediction)

# ======================================================
# METRICS
# ======================================================
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
far = fp / (fp + tn + 1e-6)
frr = fn / (fn + tp + 1e-6)

print("\n📊 AUTHENTICATION RESULTS")
print("-------------------------")
print(f"Accuracy : {accuracy:.3f}")
print(f"FAR      : {far:.3f}")
print(f"FRR      : {frr:.3f}")
print(f"Avg Sim  : {np.mean(scores):.3f}")
print("Min sim:", min(scores))
print("Max sim:", max(scores))
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
# import numpy as np
# import cv2
# from mtcnn import MTCNN
# from facenet_pytorch import InceptionResnetV1
# from sklearn.metrics import confusion_matrix

# from networks.hiding_network import HidingNetwork
# from networks.extracting_network import ExtractingNetwork
# from utils.random_matrix import generate_random_matrix

# # ======================================================
# # CONFIGURATION
# # ======================================================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FACE_FOLDER = "../datasets/celeba/img_align_celeba/img_align_celeba"
# COVER_IMAGE = "../datasets/covers/cover.jpg"
# REFERENCE_FEATURE = "../datasets/processed_faces/reference_feature.npy"

# NUM_TEST_IMAGES = 50   # evaluation size
# THRESHOLDS = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

# # ======================================================
# # LOAD MODELS
# # ======================================================
# print("🔹 Loading models...")

# hn = HidingNetwork().to(DEVICE)
# hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
# hn.eval()

# en = ExtractingNetwork().to(DEVICE)
# en.load_state_dict(torch.load("../models/extracting_network_dataset.pth"))
# en.eval()

# detector = MTCNN()
# R = generate_random_matrix().to(DEVICE)

# # ======================================================
# # HELPER FUNCTIONS
# # ======================================================
# def cosine_similarity(a, b):
#     a = a / torch.norm(a)
#     b = b / torch.norm(b)
#     return torch.dot(a, b).item()

# def preprocess_face(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         return None

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(img)
#     if len(faces) == 0:
#         return None

#     x, y, w, h = faces[0]["box"]
#     face = img[y:y+h, x:x+w]
#     face = cv2.resize(face, (256, 256))
#     face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
#     return face.unsqueeze(0).to(DEVICE)

# def load_cover():
#     cover = cv2.imread(COVER_IMAGE)
#     if cover is None:
#         raise ValueError("Cover image not found")

#     cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
#     cover = cv2.resize(cover, (256, 256))
#     cover = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
#     return cover.unsqueeze(0).to(DEVICE)

# # ======================================================
# # LOAD DATA
# # ======================================================
# face_files = sorted(os.listdir(FACE_FOLDER))[:NUM_TEST_IMAGES]
# reference = torch.tensor(np.load(REFERENCE_FEATURE)).to(DEVICE)
# cover = load_cover()

# scores = []
# y_true = []

# print("🔹 Extracting features for evaluation...")

# for i, img_name in enumerate(face_files):
#     img_path = os.path.join(FACE_FOLDER, img_name)
#     face = preprocess_face(img_path)

#     if face is None:
#         continue

#     with torch.no_grad():
#         stego = hn(face, cover)
#         feat = en(stego) @ R

#     score = cosine_similarity(feat.squeeze(), reference.squeeze())
#     scores.append(score)

#     # First half = genuine, second half = impostor (simulated)
#     label = 1 if i < NUM_TEST_IMAGES // 2 else 0
#     y_true.append(label)

# scores = np.array(scores)
# y_true = np.array(y_true)

# print("\n📈 SIMILARITY STATISTICS")
# print("-----------------------")
# print(f"Min similarity : {scores.min():.4f}")
# print(f"Max similarity : {scores.max():.4f}")
# print(f"Avg similarity : {scores.mean():.4f}")

# # ======================================================
# # THRESHOLD ANALYSIS
# # ======================================================
# print("\n📊 AUTHENTICATION RESULTS (THRESHOLD SWEEP)")
# print("-----------------------------------------")

# for threshold in THRESHOLDS:
#     y_pred = (scores >= threshold).astype(int)

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     far = fp / (fp + tn + 1e-6)
#     frr = fn / (fn + tp + 1e-6)

#     print(
#         f"Threshold={threshold:.2f} | "
#         f"Accuracy={accuracy:.3f} | "
#         f"FAR={far:.3f} | "
#         f"FRR={frr:.3f}"
#     )
