import sqlite3
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# -----------------------------------
# DATABASE PATH (ABSOLUTE & SAFE)
# -----------------------------------
DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "backend",
        "database",
        "app.db"
    )
)

def retrieve_and_decrypt_stego(user_id, key):
    """
    Retrieves encrypted stego image from SQLite
    and decrypts it using AES-256-GCM
    (Design A: single encrypted blob)
    """

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT stego_image_encrypted, encryption_iv, encryption_tag
            FROM templates
            WHERE user_id = ? AND status = 'ACTIVE'
        """, (user_id,))

        row = cursor.fetchone()

        if row is None:
            raise ValueError("No template found")

        encrypted_blob, encryption_iv, _ = row  # encryption_tag ignored (Design A)

        aesgcm = AESGCM(key)
        stego_bytes = aesgcm.decrypt(
            encryption_iv,
            encrypted_blob,
            None
        )

        return stego_bytes

    finally:
        conn.close()   # ✅ CRITICAL
