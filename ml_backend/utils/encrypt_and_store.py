import sqlite3
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "backend", "database", "app.db")

def encrypt_and_store_stego(
    user_id,
    stego_png_bytes,
    psnr,
    ssim,
    cover_hash
):
    # Generate AES-256 key (store securely in env in real system)
    key = AESGCM.generate_key(bit_length=256)
    aesgcm = AESGCM(key)

    iv = os.urandom(12)
    encrypted = aesgcm.encrypt(iv, stego_png_bytes, None)

    ciphertext = encrypted[:-16]
    tag = encrypted[-16:]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO templates (
            user_id,
            stego_image_encrypted,
            encryption_iv,
            encryption_tag,
            feature_extractor,
            cover_image_hash,
            quality_psnr,
            quality_ssim
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        ciphertext,
        iv,
        tag,
        "ExtractingNetwork",
        cover_hash,
        psnr,
        ssim
    ))

    conn.commit()
    conn.close()

    return key  # key stored securely per user (not DB)
