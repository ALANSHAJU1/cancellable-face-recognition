import os
import pickle

KEY_DIR = os.path.join(os.path.dirname(__file__), "..", "keys")
os.makedirs(KEY_DIR, exist_ok=True)

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
