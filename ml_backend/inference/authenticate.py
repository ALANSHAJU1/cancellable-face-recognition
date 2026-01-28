# ml_backend/inference/authenticate.py
import sys, os, json, torch, numpy as np, cv2
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ML_BACKEND_DIR)

from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.retrieve_and_decrypt import retrieve_and_decrypt_stego
from utils.key_store import load_user_key, load_user_R

def reject(reason):
    print(json.dumps({"decision": "REJECT", "reason": reason}))
    sys.stdout.flush()
    sys.exit(0)

user_id = sys.argv[1].strip().lower()
live_face_path = sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Load models ------------------
hn = HidingNetwork().to(device)
hn.load_state_dict(torch.load(
    os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
    map_location=device
))
hn.eval()

en = ExtractingNetwork().to(device)
en.load_state_dict(torch.load(
    os.path.join(ML_BACKEND_DIR, "models", "extracting_network_dataset.pth"),
    map_location=device
))
en.eval()

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ------------------ Load secrets ------------------
try:
    key = load_user_key(user_id)
    R = torch.tensor(load_user_R(user_id)).to(device)
except:
    reject("User revoked or not enrolled")

# ------------------ Face ------------------
img = cv2.imread(live_face_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = MTCNN().detect_faces(img)
if not faces:
    reject("No face detected")

x, y, w, h = faces[0]["box"]
face_crop = cv2.resize(img[y:y+h, x:x+w], (256,256))
face_tensor = torch.tensor(face_crop).permute(2,0,1).float()/255.0
face_tensor = face_tensor.unsqueeze(0).to(device)

# ------------------ IDENTITY CHECK (CRITICAL) ------------------
identity_path = os.path.join(
    ML_BACKEND_DIR, "keys", f"{user_id}_identity.npy"
)
if not os.path.exists(identity_path):
    reject("Identity template missing")

stored_identity = torch.tensor(
    np.load(identity_path)
).to(device)

with torch.no_grad():
    live_identity = facenet(face_tensor)

identity_score = torch.nn.functional.cosine_similarity(
    live_identity.squeeze(),
    stored_identity.squeeze(),
    dim=0
).item()

if identity_score < 0.6:
    reject("Face identity mismatch")

# ------------------ Stego comparison ------------------
cover = cv2.imread(os.path.join(
    ML_BACKEND_DIR, "datasets", "covers", "auth_cover.jpg"
))
cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
cover = cv2.resize(cover, (256,256))
cover_tensor = torch.tensor(cover).permute(2,0,1).float()/255.0
cover_tensor = cover_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    session_stego = hn(face_tensor, cover_tensor)

stored_bytes = retrieve_and_decrypt_stego(user_id, key)
stored_np = cv2.imdecode(np.frombuffer(stored_bytes,np.uint8),cv2.IMREAD_COLOR)
stored_np = cv2.cvtColor(stored_np, cv2.COLOR_BGR2RGB)
stored_np = cv2.resize(stored_np,(256,256))
stored_tensor = torch.tensor(stored_np).permute(2,0,1).float()/255.0
stored_tensor = stored_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    f1 = en(session_stego) @ R
    f2 = en(stored_tensor) @ R

stego_score = torch.nn.functional.cosine_similarity(
    f1.squeeze(), f2.squeeze(), dim=0
).item()

decision = "ACCEPT" if stego_score >= 0.18 else "REJECT"

print(json.dumps({
    "user_id": user_id,
    "identity_score": round(identity_score,4),
    "stego_score": round(stego_score,4),
    "decision": decision
}))
sys.stdout.flush()








# # ml_backend/inference/authenticate.py
# import sys
# import os
# import json
# import torch
# import numpy as np
# import cv2
# from mtcnn import MTCNN

# # -----------------------------------
# # PATH FIX
# # -----------------------------------
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(ML_BACKEND_DIR)

# from networks.hiding_network import HidingNetwork
# from networks.extracting_network import ExtractingNetwork
# from utils.retrieve_and_decrypt import retrieve_and_decrypt_stego
# from utils.key_store import load_user_key, load_user_R

# # -----------------------------------
# # SAFE REJECT (ALWAYS FLUSH)
# # -----------------------------------
# def reject(reason):
#     print(json.dumps({
#         "decision": "REJECT",
#         "reason": reason
#     }))
#     sys.stdout.flush()
#     sys.exit(0)

# # -----------------------------------
# # INPUT
# # -----------------------------------
# user_id = sys.argv[1].strip().lower()
# live_face_path = sys.argv[2]

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # -----------------------------------
# # LOAD MODELS
# # -----------------------------------
# hn = HidingNetwork().to(device)
# hn.load_state_dict(torch.load(
#     os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
#     map_location=device
# ))
# hn.eval()

# en = ExtractingNetwork().to(device)
# en.load_state_dict(torch.load(
#     os.path.join(ML_BACKEND_DIR, "models", "extracting_network_dataset.pth"),
#     map_location=device
# ))
# en.eval()

# # -----------------------------------
# # LOAD USER SECRETS (FAIL FAST)
# # -----------------------------------
# try:
#     key = load_user_key(user_id)
#     R = torch.tensor(load_user_R(user_id)).to(device)
# except Exception:
#     reject("User revoked or not enrolled")

# # -----------------------------------
# # FACE DETECTION
# # -----------------------------------
# detector = MTCNN()

# img = cv2.imread(live_face_path)
# if img is None:
#     reject("Live face image missing")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# faces = detector.detect_faces(img)

# if not faces:
#     reject("No face detected")

# h_img, w_img, _ = img.shape
# x, y, w, h = faces[0]["box"]

# # 🔒 Clamp bounding box
# x = max(0, x)
# y = max(0, y)
# w = min(w, w_img - x)
# h = min(h, h_img - y)

# face_crop = cv2.resize(img[y:y+h, x:x+w], (256, 256))

# face_tensor = (
#     torch.tensor(face_crop)
#     .permute(2, 0, 1)
#     .float() / 255.0
# ).unsqueeze(0).to(device)

# # -----------------------------------
# # AUTH COVER
# # -----------------------------------
# cover_path = os.path.join(
#     ML_BACKEND_DIR, "datasets", "covers", "auth_cover.jpg"
# )

# if not os.path.exists(cover_path):
#     reject("Authentication cover image missing")

# cover = cv2.imread(cover_path)
# cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
# cover = cv2.resize(cover, (256, 256))

# cover_tensor = (
#     torch.tensor(cover)
#     .permute(2, 0, 1)
#     .float() / 255.0
# ).unsqueeze(0).to(device)

# # -----------------------------------
# # SESSION STEGO
# # -----------------------------------
# with torch.no_grad():
#     session_stego = hn(face_tensor, cover_tensor)

# # -----------------------------------
# # STORED STEGO
# # -----------------------------------
# try:
#     stored_bytes = retrieve_and_decrypt_stego(user_id, key)
# except Exception:
#     reject("Template revoked or missing")

# stored_np = cv2.imdecode(
#     np.frombuffer(stored_bytes, np.uint8),
#     cv2.IMREAD_COLOR
# )
# stored_np = cv2.cvtColor(stored_np, cv2.COLOR_BGR2RGB)
# stored_np = cv2.resize(stored_np, (256, 256))

# stored_tensor = (
#     torch.tensor(stored_np)
#     .permute(2, 0, 1)
#     .float() / 255.0
# ).unsqueeze(0).to(device)

# # -----------------------------------
# # FEATURE EXTRACTION
# # -----------------------------------
# with torch.no_grad():
#     f1 = en(session_stego) @ R
#     f2 = en(stored_tensor) @ R

# score = torch.nn.functional.cosine_similarity(
#     f1.squeeze(), f2.squeeze(), dim=0
# ).item()

# decision = "ACCEPT" if score >= 0.18 else "REJECT"

# print(json.dumps({
#     "user_id": user_id,
#     "score": round(score, 4),
#     "decision": decision
# }))
# sys.stdout.flush()








