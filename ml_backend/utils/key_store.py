import os
import pickle
import numpy as np

# -----------------------------------
# ABSOLUTE KEY DIRECTORY (FINAL FIX)
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go from: ml_backend/utils → project root → keys
KEY_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "keys")
)

os.makedirs(KEY_DIR, exist_ok=True)

# -----------------------------------
# ENCRYPTION KEY STORAGE
# -----------------------------------
def save_user_key(user_id, key):
    """
    Stores encryption key securely outside the database
    """
    key_path = os.path.join(KEY_DIR, f"{user_id}.key")
    with open(key_path, "wb") as f:
        pickle.dump(key, f)

def load_user_key(user_id):
    """
    Loads encryption key during authentication
    """
    key_path = os.path.join(KEY_DIR, f"{user_id}.key")
    if not os.path.exists(key_path):
        raise FileNotFoundError("Encryption key not found for user")
    with open(key_path, "rb") as f:
        return pickle.load(f)

# -----------------------------------
# USER-SPECIFIC RANDOM MATRIX (R)
# -----------------------------------
def save_user_R(user_id, R):
    """
    Stores user-specific random projection matrix
    (cancellable biometric transformation)
    """
    R_path = os.path.join(KEY_DIR, f"{user_id}_R.npy")
    np.save(R_path, R)

def load_user_R(user_id):
    """
    Loads user-specific random projection matrix during authentication
    """
    R_path = os.path.join(KEY_DIR, f"{user_id}_R.npy")
    if not os.path.exists(R_path):
        raise FileNotFoundError("User transformation matrix not found")
    return np.load(R_path)
