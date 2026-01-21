import sys
import os

# -------------------------------
# FIX PYTHON MODULE PATH (FINAL)
# -------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ML_BACKEND_DIR)

import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

from networks.hiding_network import HidingNetwork
from utils.encrypt_and_store import encrypt_and_store_stego
from utils.key_store import save_user_key, save_user_R
from utils.random_matrix import generate_random_matrix

# -------------------------------
# INPUT ARGUMENTS (NORMALIZED)
# -------------------------------
username = sys.argv[1].strip().lower()
face_image_path = sys.argv[2]
cover_image_path = sys.argv[3]

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# LOAD MODELS
# -------------------------------
hn = HidingNetwork().to(device)
hn.load_state_dict(
    torch.load(
        os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
        map_location=device
    )
)
hn.eval()

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# -------------------------------
# USER-SPECIFIC RANDOM MATRIX
# -------------------------------
R = generate_random_matrix().cpu().numpy()
save_user_R(username, R)

# -------------------------------
# FACE DETECTION
# -------------------------------
detector = MTCNN()

face_img = cv2.imread(face_image_path)
if face_img is None:
    raise ValueError("Face image not found")

face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(face_img)
if len(faces) == 0:
    raise ValueError("No face detected")

x, y, w, h = faces[0]["box"]
face_crop = cv2.resize(face_img[y:y+h, x:x+w], (256, 256))

face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
face_tensor = face_tensor.unsqueeze(0).to(device)

# -------------------------------
# SAVE FACENET REFERENCE (CRITICAL)
# -------------------------------
with torch.no_grad():
    face_embedding = facenet(face_tensor).cpu().numpy()

ref_dir = os.path.join(ML_BACKEND_DIR, "datasets", "processed_faces")
os.makedirs(ref_dir, exist_ok=True)

np.save(
    os.path.join(ref_dir, f"{username}_reference.npy"),
    face_embedding
)

# -------------------------------
# COVER IMAGE
# -------------------------------
cover_img = cv2.imread(cover_image_path)
if cover_img is None:
    raise ValueError("Cover image not found")

cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
cover_img = cv2.resize(cover_img, (256, 256))

cover_tensor = torch.tensor(cover_img).permute(2, 0, 1).float() / 255.0
cover_tensor = cover_tensor.unsqueeze(0).to(device)

# -------------------------------
# GENERATE STEGO
# -------------------------------
with torch.no_grad():
    stego_tensor = hn(face_tensor, cover_tensor)

stego_np = (
    stego_tensor.squeeze(0)
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)
stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

_, stego_png = cv2.imencode(".png", stego_np)

# -------------------------------
# STORE ENCRYPTED TEMPLATE
# -------------------------------
key = encrypt_and_store_stego(
    user_id=username,
    stego_png_bytes=stego_png.tobytes(),
    psnr=38.0,
    ssim=0.96,
    cover_hash=os.path.basename(cover_image_path)
)

save_user_key(username, key)

print(f"Enrollment completed successfully for user: {username}")



# import sys
# import os

# # -------------------------------
# # FIX PYTHON MODULE PATH (FINAL)
# # -------------------------------
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(ML_BACKEND_DIR)

# import cv2
# import torch
# import numpy as np
# from mtcnn import MTCNN

# from networks.hiding_network import HidingNetwork
# from utils.encrypt_and_store import encrypt_and_store_stego
# from utils.key_store import save_user_key, save_user_R      # ✅ CHANGE
# from utils.random_matrix import generate_random_matrix      # ✅ CHANGE

# # -------------------------------
# # INPUT ARGUMENTS FROM NODE.JS
# # -------------------------------
# username = sys.argv[1]
# face_image_path = sys.argv[2]
# cover_image_path = sys.argv[3]

# # -------------------------------
# # DEVICE
# # -------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # -------------------------------
# # LOAD HIDING NETWORK
# # -------------------------------
# hn = HidingNetwork().to(device)

# model_path = os.path.join(
#     ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"
# )

# hn.load_state_dict(torch.load(model_path, map_location=device))
# hn.eval()

# # -------------------------------
# # GENERATE & STORE USER-SPECIFIC RANDOM MATRIX (CRITICAL FIX)
# # -------------------------------
# R = generate_random_matrix().cpu().numpy()
# save_user_R(username, R)

# # -------------------------------
# # FACE DETECTOR
# # -------------------------------
# detector = MTCNN()

# # -------------------------------
# # LOAD & PREPROCESS FACE IMAGE
# # -------------------------------
# face_img = cv2.imread(face_image_path)
# if face_img is None:
#     raise ValueError("Face image not found")

# face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
# faces = detector.detect_faces(face_img)

# if len(faces) == 0:
#     raise ValueError("No face detected in face image")

# x, y, w, h = faces[0]["box"]
# face_crop = face_img[y:y+h, x:x+w]
# face_crop = cv2.resize(face_crop, (256, 256))

# face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
# face_tensor = face_tensor.unsqueeze(0).to(device)

# # -------------------------------
# # LOAD & PREPROCESS COVER IMAGE
# # -------------------------------
# cover_img = cv2.imread(cover_image_path)
# if cover_img is None:
#     raise ValueError("Cover image not found")

# cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
# cover_img = cv2.resize(cover_img, (256, 256))

# cover_tensor = torch.tensor(cover_img).permute(2, 0, 1).float() / 255.0
# cover_tensor = cover_tensor.unsqueeze(0).to(device)

# # -------------------------------
# # GENERATE STEGO IMAGE (HN)
# # -------------------------------
# with torch.no_grad():
#     stego_tensor = hn(face_tensor, cover_tensor)

# # -------------------------------
# # CONVERT STEGO TO PNG BYTES
# # -------------------------------
# stego_np = (
#     stego_tensor.squeeze(0)
#     .permute(1, 2, 0)
#     .cpu()
#     .numpy()
# )
# stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

# success, stego_png = cv2.imencode(".png", stego_np)
# if not success:
#     raise ValueError("Failed to encode stego image")

# stego_png_bytes = stego_png.tobytes()

# # -------------------------------
# # QUALITY METRICS (STATIC / REPORT)
# # -------------------------------
# psnr = 38.0
# ssim = 0.96
# cover_hash = os.path.basename(cover_image_path)

# # -------------------------------
# # ENCRYPT & STORE IN SQLITE
# # -------------------------------
# key = encrypt_and_store_stego(
#     user_id=username,
#     stego_png_bytes=stego_png_bytes,
#     psnr=psnr,
#     ssim=ssim,
#     cover_hash=cover_hash
# )

# # Save encryption key securely (outside DB)
# save_user_key(username, key)

# print(f"Enrollment completed successfully for user: {username}")
