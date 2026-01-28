# ml_backend/inference/enroll.py
import sys, os, sqlite3, traceback
import cv2, torch, numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ML_BACKEND_DIR)

from networks.hiding_network import HidingNetwork
from utils.encrypt_and_store import encrypt_and_store_stego
from utils.key_store import save_user_key, save_user_R
from utils.random_matrix import generate_random_matrix

def main():
    username = sys.argv[1].strip().lower()
    face_image_path = sys.argv[2]
    cover_image_path = sys.argv[3]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ Revoke old template ------------------
    DB_PATH = os.path.join(ML_BACKEND_DIR, "..", "backend", "database", "app.db")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE templates SET status='REVOKED' WHERE user_id=? AND status='ACTIVE'",
        (username,)
    )
    conn.commit()
    conn.close()

    # ------------------ Load models ------------------
    hn = HidingNetwork().to(device)
    hn.load_state_dict(torch.load(
        os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
        map_location=device
    ))
    hn.eval()

    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # ------------------ Face detection ------------------
    img = cv2.imread(face_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = MTCNN().detect_faces(img)
    if not faces:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]["box"]
    face_crop = cv2.resize(img[y:y+h, x:x+w], (256, 256))

    face_tensor = torch.tensor(face_crop).permute(2,0,1).float()/255.0
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # ------------------ Save identity embedding ------------------
    with torch.no_grad():
        identity_emb = facenet(face_tensor).cpu().numpy()

    identity_path = os.path.join(
        ML_BACKEND_DIR, "keys", f"{username}_identity.npy"
    )
    np.save(identity_path, identity_emb)

    # ------------------ Cancellable template ------------------
    R = generate_random_matrix().cpu().numpy()
    save_user_R(username, R)

    cover = cv2.imread(cover_image_path)
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover = cv2.resize(cover, (256,256))
    cover_tensor = torch.tensor(cover).permute(2,0,1).float()/255.0
    cover_tensor = cover_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        stego = hn(face_tensor, cover_tensor)

    stego_np = stego.squeeze(0).permute(1,2,0).cpu().numpy()
    stego_np = np.clip(stego_np*255, 0, 255).astype(np.uint8)
    _, stego_png = cv2.imencode(".png", stego_np)

    key = encrypt_and_store_stego(
        username,
        stego_png.tobytes(),
        psnr=38.0,
        ssim=0.96,
        cover_hash=os.path.basename(cover_image_path)
    )

    save_user_key(username, key)
    print("Enrollment successful")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ENROLL ERROR:", e)
        traceback.print_exc()
        sys.exit(1)




# import sys
# import os
# import sqlite3
# import traceback

# print("ENROLL SCRIPT STARTED")
# print("ARGV:", sys.argv)

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(ML_BACKEND_DIR)

# import cv2
# import torch
# import numpy as np
# from mtcnn import MTCNN
# from facenet_pytorch import InceptionResnetV1

# from networks.hiding_network import HidingNetwork
# from utils.encrypt_and_store import encrypt_and_store_stego
# from utils.key_store import save_user_key, save_user_R
# from utils.random_matrix import generate_random_matrix


# def main():
#     if len(sys.argv) < 4:
#         raise ValueError("Missing arguments")

#     username = sys.argv[1].strip().lower()
#     face_image_path = os.path.normpath(sys.argv[2])
#     cover_image_path = os.path.normpath(sys.argv[3])

#     print("Username:", username)
#     print("Face path:", face_image_path)
#     print("Cover path:", cover_image_path)
#     print("Face exists:", os.path.exists(face_image_path))
#     print("Cover exists:", os.path.exists(cover_image_path))

#     if not os.path.exists(face_image_path):
#         raise ValueError("Face image file does not exist")

#     if not os.path.exists(cover_image_path):
#         raise ValueError("Cover image file does not exist")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Device:", device)

#     DB_PATH = os.path.abspath(
#         os.path.join(ML_BACKEND_DIR, "..", "backend", "database", "app.db")
#     )

#     print("Revoking old templates")
#     conn = sqlite3.connect(DB_PATH, timeout=30)
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             UPDATE templates
#             SET status = 'REVOKED'
#             WHERE user_id = ? AND status = 'ACTIVE'
#         """, (username,))
#         conn.commit()
#         print("Old templates revoked:", cursor.rowcount)
#     finally:
#         conn.close()

#     print("Loading models")
#     hn = HidingNetwork().to(device)
#     hn.load_state_dict(
#         torch.load(
#             os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
#             map_location=device
#         )
#     )
#     hn.eval()

#     facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

#     print("Generating random matrix")
#     R = generate_random_matrix().cpu().numpy()
#     save_user_R(username, R)

#     print("Detecting face")
#     detector = MTCNN()

#     face_img = cv2.imread(face_image_path)
#     if face_img is None:
#         raise ValueError("OpenCV failed to read face image")

#     face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(face_img)
#     if len(faces) == 0:
#         raise ValueError("No face detected")

#     x, y, w, h = faces[0]["box"]
#     face_crop = cv2.resize(face_img[y:y+h, x:x+w], (256, 256))

#     face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
#     face_tensor = face_tensor.unsqueeze(0).to(device)

#     print("Saving FaceNet reference")
#     with torch.no_grad():
#         face_embedding = facenet(face_tensor).cpu().numpy()

#     ref_dir = os.path.join(ML_BACKEND_DIR, "datasets", "processed_faces")
#     os.makedirs(ref_dir, exist_ok=True)

#     np.save(os.path.join(ref_dir, f"{username}_reference.npy"), face_embedding)

#     print("Processing cover image")
#     cover_img = cv2.imread(cover_image_path)
#     if cover_img is None:
#         raise ValueError("OpenCV failed to read cover image")

#     cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
#     cover_img = cv2.resize(cover_img, (256, 256))

#     cover_tensor = torch.tensor(cover_img).permute(2, 0, 1).float() / 255.0
#     cover_tensor = cover_tensor.unsqueeze(0).to(device)

#     print("Generating stego")
#     with torch.no_grad():
#         stego_tensor = hn(face_tensor, cover_tensor)

#     stego_np = stego_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

#     success, stego_png = cv2.imencode(".png", stego_np)
#     if not success:
#         raise ValueError("Failed to encode stego")

#     print("Storing encrypted template")
#     key = encrypt_and_store_stego(
#         user_id=username,
#         stego_png_bytes=stego_png.tobytes(),
#         psnr=38.0,
#         ssim=0.96,
#         cover_hash=os.path.basename(cover_image_path)
#     )

#     save_user_key(username, key)

#     print("Enrollment completed successfully for user:", username)


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print("ENROLLMENT ERROR:", str(e))
#         traceback.print_exc()
#         sys.exit(1)
