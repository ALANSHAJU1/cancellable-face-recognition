import sqlite3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DB_PATH = "../backend/database/app.db"

def retrieve_and_decrypt_stego(user_id, key):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT stego_image_encrypted, encryption_iv, encryption_tag
        FROM templates
        WHERE user_id = ? AND status = 'ACTIVE'
    """, (user_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise ValueError("No template found")

    ciphertext, iv, tag = row
    aesgcm = AESGCM(key)

    encrypted = ciphertext + tag
    stego_bytes = aesgcm.decrypt(iv, encrypted, None)

    return stego_bytes
