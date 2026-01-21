import sys
import os
import json
import torch
import numpy as np
import cv2
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

# -------------------------------
# FIX PYTHON MODULE PATH
# -------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ML_BACKEND_DIR)

from utils.retrieve_and_decrypt import retrieve_and_decrypt_stego
from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.key_store import load_user_key, load_user_R

# -----------------------------
# INPUT (NORMALIZED)
# -----------------------------
user_id = sys.argv[1].strip().lower()
live_face_path = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODELS
# -----------------------------
hn = HidingNetwork().to(device)
hn.load_state_dict(
    torch.load(
        os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
        map_location=device
    )
)
hn.eval()

en = ExtractingNetwork().to(device)
en.load_state_dict(
    torch.load(
        os.path.join(ML_BACKEND_DIR, "models", "extracting_network_dataset.pth"),
        map_location=device
    )
)
en.eval()

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

R = torch.tensor(load_user_R(user_id)).to(device)
detector = MTCNN()

# -----------------------------
# LIVE FACE
# -----------------------------
img = cv2.imread(live_face_path)
if img is None:
    raise ValueError("Live face image not found")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(img)
if len(faces) == 0:
    raise ValueError("No face detected")

x, y, w, h = faces[0]["box"]
face_crop = cv2.resize(img[y:y+h, x:x+w], (256, 256))

face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
face_tensor = face_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    live_face_emb = facenet(face_tensor)

# -----------------------------
# LOAD STORED FACENET REFERENCE
# -----------------------------
ref_path = os.path.join(
    ML_BACKEND_DIR,
    "datasets",
    "processed_faces",
    f"{user_id}_reference.npy"
)

if not os.path.exists(ref_path):
    raise ValueError("Reference face embedding not found")

reference_face_emb = torch.tensor(np.load(ref_path)).to(device)

# -----------------------------
# FACE IDENTITY CHECK
# -----------------------------
def cosine_similarity(a, b):
    a = a / torch.norm(a)
    b = b / torch.norm(b)
    return torch.dot(a, b).item()

face_score = cosine_similarity(
    live_face_emb.squeeze(),
    reference_face_emb.squeeze()
)

# -----------------------------
# FINAL DECISION
# -----------------------------
FACE_THRESHOLD = 0.6

decision = "ACCEPT" if face_score >= FACE_THRESHOLD else "REJECT"

print(json.dumps({
    "user_id": user_id,
    "face_score": round(face_score, 4),
    "decision": decision
}))





# import sys
# import os
# import json
# import torch
# import numpy as np
# import cv2
# from mtcnn import MTCNN
# from facenet_pytorch import InceptionResnetV1

# # -------------------------------
# # FIX PYTHON MODULE PATH
# # -------------------------------
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(ML_BACKEND_DIR)

# from utils.retrieve_and_decrypt import retrieve_and_decrypt_stego
# from networks.hiding_network import HidingNetwork
# from networks.extracting_network import ExtractingNetwork
# from utils.key_store import load_user_key, load_user_R

# # -----------------------------
# # INPUT (NORMALIZED)
# # -----------------------------
# user_id = sys.argv[1].strip().lower()
# live_face_path = sys.argv[2]

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # -----------------------------
# # LOAD MODELS
# # -----------------------------
# hn = HidingNetwork().to(device)
# hn.load_state_dict(
#     torch.load(
#         os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
#         map_location=device
#     )
# )
# hn.eval()

# en = ExtractingNetwork().to(device)
# en.load_state_dict(
#     torch.load(
#         os.path.join(ML_BACKEND_DIR, "models", "extracting_network_dataset.pth"),
#         map_location=device
#     )
# )
# en.eval()

# # Face identity model
# facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# # -----------------------------
# # LOAD USER-SPECIFIC RANDOM MATRIX
# # -----------------------------
# R = torch.tensor(load_user_R(user_id)).to(device)
# detector = MTCNN()

# # -----------------------------
# # LOAD LIVE FACE IMAGE
# # -----------------------------
# face_img = cv2.imread(live_face_path)
# if face_img is None:
#     raise ValueError("Live face image not found")

# face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
# faces = detector.detect_faces(face_img)

# if len(faces) == 0:
#     raise ValueError("No face detected in live image")

# x, y, w, h = faces[0]["box"]
# face_crop = cv2.resize(face_img[y:y+h, x:x+w], (256, 256))

# face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
# face_tensor = face_tensor.unsqueeze(0).to(device)

# # -----------------------------
# # LIVE FACE EMBEDDING (FaceNet)
# # -----------------------------
# with torch.no_grad():
#     live_face_emb = facenet(face_tensor)

# # -----------------------------
# # LOAD AUTH COVER IMAGE
# # -----------------------------
# cover_path = os.path.join(
#     ML_BACKEND_DIR, "datasets", "covers", "auth_cover.jpg"
# )

# cover_img = cv2.imread(cover_path)
# if cover_img is None:
#     raise ValueError("Authentication cover image not found")

# cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
# cover_img = cv2.resize(cover_img, (256, 256))

# cover_tensor = torch.tensor(cover_img).permute(2, 0, 1).float() / 255.0
# cover_tensor = cover_tensor.unsqueeze(0).to(device)

# # -----------------------------
# # CREATE SESSION STEGO
# # -----------------------------
# with torch.no_grad():
#     session_stego = hn(face_tensor, cover_tensor)

# # -----------------------------
# # LOAD STORED STEGO FROM DB
# # -----------------------------
# key = load_user_key(user_id)
# stored_stego_bytes = retrieve_and_decrypt_stego(user_id, key)

# stored_stego_np = cv2.imdecode(
#     np.frombuffer(stored_stego_bytes, np.uint8),
#     cv2.IMREAD_COLOR
# )
# stored_stego_np = cv2.cvtColor(stored_stego_np, cv2.COLOR_BGR2RGB)
# stored_stego_np = cv2.resize(stored_stego_np, (256, 256))

# stored_stego_tensor = torch.tensor(stored_stego_np).permute(2, 0, 1).float() / 255.0
# stored_stego_tensor = stored_stego_tensor.unsqueeze(0).to(device)

# # -----------------------------
# # EXTRACT STEGO FEATURES (EN)
# # -----------------------------
# with torch.no_grad():
#     feat_session = en(session_stego) @ R
#     feat_stored = en(stored_stego_tensor) @ R

# # -----------------------------
# # LOAD STORED FACENET REFERENCE
# # -----------------------------
# reference_path = os.path.join(
#     ML_BACKEND_DIR,
#     "datasets",
#     "processed_faces",
#     f"{user_id}_reference.npy"
# )

# if not os.path.exists(reference_path):
#     raise ValueError("Reference face embedding not found")

# reference_face_emb = torch.tensor(
#     np.load(reference_path)
# ).to(device)

# # -----------------------------
# # SIMILARITY
# # -----------------------------
# def cosine_similarity(a, b):
#     a = a / torch.norm(a)
#     b = b / torch.norm(b)
#     return torch.dot(a, b).item()

# stego_score = cosine_similarity(
#     feat_session.squeeze(),
#     feat_stored.squeeze()
# )

# face_score = cosine_similarity(
#     live_face_emb.squeeze(),
#     reference_face_emb.squeeze()
# )

# # -----------------------------
# # FINAL DECISION
# # -----------------------------
# STEGO_THRESHOLD = 0.18
# FACE_THRESHOLD = 0.6

# if stego_score >= STEGO_THRESHOLD and face_score >= FACE_THRESHOLD:
#     decision = "ACCEPT"
# else:
#     decision = "REJECT"

# # -----------------------------
# # OUTPUT
# # -----------------------------
# print(json.dumps({
#     "user_id": user_id,
#     "stego_score": round(stego_score, 4),
#     "face_score": round(face_score, 4),
#     "decision": decision
# }))
