import os
import sys
import sqlite3

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "backend", "database", "app.db")
KEY_DIR = os.path.join(BASE_DIR, "ml_backend", "keys")

user_id = sys.argv[1]

# -------------------------------
# 1. REVOKE TEMPLATE IN DATABASE
# -------------------------------
conn = sqlite3.connect(DB_PATH, timeout=30)
try:
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE templates
        SET status = 'REVOKED'
        WHERE user_id = ? AND status = 'ACTIVE'
    """, (user_id,))
    conn.commit()
finally:
    conn.close()

# -------------------------------
# 2. DELETE USER KEYS
# -------------------------------
key_file = os.path.join(KEY_DIR, f"{user_id}.key")
R_file = os.path.join(KEY_DIR, f"{user_id}_R.npy")

if os.path.exists(key_file):
    os.remove(key_file)

if os.path.exists(R_file):
    os.remove(R_file)

print(f"Revocation completed for user: {user_id}")
