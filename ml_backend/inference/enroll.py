# ml_backend/inference/enroll.py

import sys, os, traceback
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


# ============================
# Centralized Model Path
# ============================
HN_MODEL_PATH = os.path.join(
    ML_BACKEND_DIR,
    "models",
    "hiding_network_dataset.pth"
)


def main():
    username = sys.argv[1].strip().lower()
    face_image_path = sys.argv[2]
    cover_image_path = sys.argv[3]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ Load HN Model ------------------

    if not os.path.exists(HN_MODEL_PATH):
        raise FileNotFoundError(f"HN model not found at {HN_MODEL_PATH}")

    hn = HidingNetwork().to(device)
    hn.load_state_dict(torch.load(HN_MODEL_PATH, map_location=device))
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

    face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # ------------------ Save identity embedding ------------------

    with torch.no_grad():
        identity_emb = facenet(face_tensor).cpu().numpy()

    identity_path = os.path.join(
        ML_BACKEND_DIR, "keys", f"{username}_identity.npy"
    )

    os.makedirs(os.path.dirname(identity_path), exist_ok=True)
    np.save(identity_path, identity_emb)

    # ------------------ Cancellable template ------------------

    R = generate_random_matrix().cpu().numpy()
    save_user_R(username, R)

    cover = cv2.imread(cover_image_path)
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    cover = cv2.resize(cover, (256, 256))

    cover_tensor = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
    cover_tensor = cover_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        stego_tensor = hn(face_tensor, cover_tensor)

    # ------------------ SAVE STEGO IMAGE ------------------

    stego_dir = os.path.join(ML_BACKEND_DIR, "datasets", "stego", "enroll")
    os.makedirs(stego_dir, exist_ok=True)

    stego_np = stego_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

    stego_path = os.path.join(stego_dir, f"{username}_stego.png")
    cv2.imwrite(stego_path, cv2.cvtColor(stego_np, cv2.COLOR_RGB2BGR))

    _, stego_png = cv2.imencode(".png", stego_np)

    key = encrypt_and_store_stego(
        username,
        stego_png.tobytes(),
        psnr=34.03,
        ssim=0.95,
        cover_hash=os.path.basename(cover_image_path)
    )

    save_user_key(username, key)

    print("Enrollment successful")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ENROLL ERROR:", str(e))
        traceback.print_exc()
        sys.exit(1)










































# import sys, os, sqlite3, traceback
# import cv2, torch, numpy as np
# from mtcnn import MTCNN
# from facenet_pytorch import InceptionResnetV1

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(ML_BACKEND_DIR)

# from networks.hiding_network import HidingNetwork
# from utils.encrypt_and_store import encrypt_and_store_stego
# from utils.key_store import save_user_key, save_user_R
# from utils.random_matrix import generate_random_matrix


# # ============================
# #  CHANGE 1: Centralized Model Path
# # ============================
# HN_MODEL_PATH = os.path.join(
#     ML_BACKEND_DIR,
#     "models",
#     "hiding_network_dataset.pth"   # <-- Your trained best model
# )


# def main():
#     username = sys.argv[1].strip().lower()
#     face_image_path = sys.argv[2]
#     cover_image_path = sys.argv[3]

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # ------------------ Revoke old template ------------------
#     DB_PATH = os.path.join(ML_BACKEND_DIR, "..", "backend", "database", "app.db")
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         "UPDATE templates SET status='REVOKED' WHERE user_id=? AND status='ACTIVE'",
#         (username,)
#     )
#     conn.commit()
#     conn.close()

#     # ------------------ Load HN Model ------------------

#     #  CHANGE 2: Safe loading
#     if not os.path.exists(HN_MODEL_PATH):
#         raise FileNotFoundError(f"HN model not found at {HN_MODEL_PATH}")

#     hn = HidingNetwork().to(device)
#     hn.load_state_dict(torch.load(HN_MODEL_PATH, map_location=device))
#     hn.eval()

#     facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

#     # ------------------ Face detection ------------------
#     img = cv2.imread(face_image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = MTCNN().detect_faces(img)

#     if not faces:
#         raise ValueError("No face detected")

#     x, y, w, h = faces[0]["box"]
#     face_crop = cv2.resize(img[y:y+h, x:x+w], (256, 256))

#     face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
#     face_tensor = face_tensor.unsqueeze(0).to(device)

#     # ------------------ Save identity embedding ------------------
#     with torch.no_grad():
#         identity_emb = facenet(face_tensor).cpu().numpy()

#     identity_path = os.path.join(
#         ML_BACKEND_DIR, "keys", f"{username}_identity.npy"
#     )
#     np.save(identity_path, identity_emb)

#     # ------------------ Cancellable template ------------------
#     R = generate_random_matrix().cpu().numpy()
#     save_user_R(username, R)

#     cover = cv2.imread(cover_image_path)
#     cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
#     cover = cv2.resize(cover, (256, 256))

#     cover_tensor = torch.tensor(cover).permute(2, 0, 1).float() / 255.0
#     cover_tensor = cover_tensor.unsqueeze(0).to(device)

#     with torch.no_grad():
#         stego_tensor = hn(face_tensor, cover_tensor)

#     # ------------------ SAVE STEGO IMAGE ------------------
#     stego_dir = os.path.join(ML_BACKEND_DIR, "datasets", "stego", "enroll")
#     os.makedirs(stego_dir, exist_ok=True)

#     stego_np = stego_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     stego_np = np.clip(stego_np * 255, 0, 255).astype(np.uint8)

#     stego_path = os.path.join(stego_dir, f"{username}_stego.png")
#     cv2.imwrite(stego_path, cv2.cvtColor(stego_np, cv2.COLOR_RGB2BGR))

#     _, stego_png = cv2.imencode(".png", stego_np)

#     key = encrypt_and_store_stego(
#         username,
#         stego_png.tobytes(),
#         psnr=34.03,   # 🔥 OPTIONAL: You can now use real computed values
#         ssim=0.95,
#         cover_hash=os.path.basename(cover_image_path)
#     )

#     save_user_key(username, key)
#     print("Enrollment successful")


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print("ENROLL ERROR:", str(e))
#         traceback.print_exc()
#         sys.exit(1)

















































# import sys, os, sqlite3, traceback
# import cv2, torch, numpy as np
# from mtcnn import MTCNN
# from facenet_pytorch import InceptionResnetV1

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(ML_BACKEND_DIR)

# from networks.hiding_network import HidingNetwork
# from utils.encrypt_and_store import encrypt_and_store_stego
# from utils.key_store import save_user_key, save_user_R
# from utils.random_matrix import generate_random_matrix

# def main():
#     username = sys.argv[1].strip().lower()
#     face_image_path = sys.argv[2]
#     cover_image_path = sys.argv[3]

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # ------------------ Revoke old template ------------------
#     DB_PATH = os.path.join(ML_BACKEND_DIR, "..", "backend", "database", "app.db")
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         "UPDATE templates SET status='REVOKED' WHERE user_id=? AND status='ACTIVE'",
#         (username,)
#     )
#     conn.commit()
#     conn.close()

#     # ------------------ Load models ------------------
#     hn = HidingNetwork().to(device)
#     hn.load_state_dict(torch.load(
#         os.path.join(ML_BACKEND_DIR, "models", "hiding_network_dataset.pth"),
#         map_location=device
#     ))
#     hn.eval()

#     facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

#     # ------------------ Face detection ------------------
#     img = cv2.imread(face_image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = MTCNN().detect_faces(img)
#     if not faces:
#         raise ValueError("No face detected")

#     x, y, w, h = faces[0]["box"]
#     face_crop = cv2.resize(img[y:y+h, x:x+w], (256, 256))

#     face_tensor = torch.tensor(face_crop).permute(2,0,1).float()/255.0
#     face_tensor = face_tensor.unsqueeze(0).to(device)

#     # ------------------ Save identity embedding ------------------
#     with torch.no_grad():
#         identity_emb = facenet(face_tensor).cpu().numpy()

#     identity_path = os.path.join(
#         ML_BACKEND_DIR, "keys", f"{username}_identity.npy"
#     )
#     np.save(identity_path, identity_emb)

#     # ------------------ Cancellable template ------------------
#     R = generate_random_matrix().cpu().numpy()
#     save_user_R(username, R)

#     cover = cv2.imread(cover_image_path)
#     cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
#     cover = cv2.resize(cover, (256,256))
#     cover_tensor = torch.tensor(cover).permute(2,0,1).float()/255.0
#     cover_tensor = cover_tensor.unsqueeze(0).to(device)

#     with torch.no_grad():
#         stego = hn(face_tensor, cover_tensor)

#     stego_np = stego.squeeze(0).permute(1,2,0).cpu().numpy()
#     stego_np = np.clip(stego_np*255, 0, 255).astype(np.uint8)
#     _, stego_png = cv2.imencode(".png", stego_np)

#     key = encrypt_and_store_stego(
#         username,
#         stego_png.tobytes(),
#         psnr=38.0,
#         ssim=0.96,
#         cover_hash=os.path.basename(cover_image_path)
#     )

#     save_user_key(username, key)
#     print("Enrollment successful")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print("ENROLL ERROR:", e)
#         traceback.print_exc()
#         sys.exit(1)














