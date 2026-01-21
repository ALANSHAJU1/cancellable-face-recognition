import sqlite3
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# -----------------------------------
# DATABASE PATH (ABSOLUTE & SAFE)
# -----------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "backend", "database", "app.db")

# -----------------------------------
# ENCRYPT & STORE STEGO TEMPLATE
# (DESIGN A — Single Encrypted Blob)
# -----------------------------------
def encrypt_and_store_stego(
    user_id,
    stego_png_bytes,
    psnr,
    ssim,
    cover_hash
):
    # -------------------------------
    # AES-256-GCM Encryption
    # -------------------------------
    key = AESGCM.generate_key(bit_length=256)
    aesgcm = AESGCM(key)

    encryption_iv = os.urandom(12)
    encrypted_blob = aesgcm.encrypt(
        encryption_iv,
        stego_png_bytes,
        None
    )  # ciphertext + tag (combined)

    # -------------------------------
    # SQLITE CONNECTION
    # -------------------------------
    conn = sqlite3.connect(DB_PATH, timeout=30)

    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO templates (
                user_id,
                stego_image_encrypted,
                encryption_iv,
                encryption_tag,
                cover_image_hash,
                quality_psnr,
                quality_ssim,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            encrypted_blob,      # ✅ combined ciphertext + tag
            encryption_iv,       # ✅ IV stored
            b"",                 # ✅ empty tag (Design A, NOT NULL safe)
            cover_hash,
            psnr,
            ssim,
            "ACTIVE"
        ))

        conn.commit()
        return key  # stored securely outside DB

    finally:
        conn.close()  # ✅ CRITICAL
