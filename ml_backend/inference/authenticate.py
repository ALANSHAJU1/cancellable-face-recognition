import sys
import json
import torch
import numpy as np
import cv2

from utils.retrieve_and_decrypt import retrieve_and_decrypt_stego
from networks.hiding_network import HidingNetwork
from networks.extracting_network import ExtractingNetwork
from utils.random_matrix import generate_random_matrix

# -----------------------------
# INPUT
# -----------------------------
user_id = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODELS
# -----------------------------
hn = HidingNetwork().to(device)
hn.load_state_dict(torch.load("../models/hiding_network_dataset.pth"))
hn.eval()

en = ExtractingNetwork().to(device)
en.load_state_dict(torch.load("../models/extracting_network_dataset.pth"))
en.eval()

R = generate_random_matrix().to(device)

# -----------------------------
# LOAD ENCRYPTED STEGO
# -----------------------------
# NOTE: key management assumed secure (as explained in report)
from utils.key_store import load_user_key
key = load_user_key(user_id)

stego_bytes = retrieve_and_decrypt_stego(user_id, key)

# Convert stego bytes → tensor
stego_np = cv2.imdecode(
    np.frombuffer(stego_bytes, np.uint8),
    cv2.IMREAD_COLOR
)
stego_np = cv2.cvtColor(stego_np, cv2.COLOR_BGR2RGB)
stego_np = cv2.resize(stego_np, (256, 256))

stego_tensor = torch.tensor(stego_np).permute(2, 0, 1).float() / 255.0
stego_tensor = stego_tensor.unsqueeze(0).to(device)

# -----------------------------
# EXTRACT FEATURE
# -----------------------------
with torch.no_grad():
    extracted_feature = en(stego_tensor)
    protected_feature = extracted_feature @ R

# -----------------------------
# LOAD REFERENCE FEATURE
# -----------------------------
reference = np.load(f"../datasets/processed_faces/{user_id}_reference.npy")
reference = torch.tensor(reference).to(device)

# -----------------------------
# SIMILARITY
# -----------------------------
def cosine_similarity(a, b):
    a = a / torch.norm(a)
    b = b / torch.norm(b)
    return torch.dot(a, b).item()

score = cosine_similarity(
    protected_feature.squeeze(),
    reference.squeeze()
)

THRESHOLD = 0.18
decision = "ACCEPT" if score >= THRESHOLD else "REJECT"

# -----------------------------
# OUTPUT
# -----------------------------
print(json.dumps({
    "user_id": user_id,
    "score": round(score, 4),
    "decision": decision
}))
