import os
import pickle

KEY_DIR = "../keys"
os.makedirs(KEY_DIR, exist_ok=True)

def load_user_key(user_id):
    with open(f"{KEY_DIR}/{user_id}.key", "rb") as f:
        return pickle.load(f)
