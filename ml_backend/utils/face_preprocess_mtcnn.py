import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import os

# -------------------------------
# CONFIGURATION (DO NOT CHANGE)
# -------------------------------
INPUT_IMAGE_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba/sample.jpg"

OUTPUT_DIR = "../datasets/processed_faces"
OUTPUT_SIZE = (256, 256)
MARGIN = 22   # as mentioned in report

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# STEP 1: Load Image
# -------------------------------
image_bgr = cv2.imread(INPUT_IMAGE_PATH)

if image_bgr is None:
    raise ValueError("Image not found or path incorrect")

# Convert BGR (OpenCV) → RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------
# STEP 2: Initialize Pretrained MTCNN
# -------------------------------
detector = MTCNN()

# -------------------------------
# STEP 3: Detect Face
# -------------------------------
results = detector.detect_faces(image_rgb)

if len(results) == 0:
    raise ValueError("No face detected in the image")

# Take the first detected face
face = results[0]
x, y, width, height = face['box']

# -------------------------------
# STEP 4: Add Margin and Crop
# -------------------------------
img_h, img_w, _ = image_rgb.shape

x1 = max(x - MARGIN, 0)
y1 = max(y - MARGIN, 0)
x2 = min(x + width + MARGIN, img_w)
y2 = min(y + height + MARGIN, img_h)

face_crop = image_rgb[y1:y2, x1:x2]

# -------------------------------
# STEP 5: Resize to 256 × 256
# -------------------------------
face_pil = Image.fromarray(face_crop)
face_resized = face_pil.resize(OUTPUT_SIZE)

# -------------------------------
# STEP 6: Normalize (0–1)
# -------------------------------
face_array = np.asarray(face_resized).astype(np.float32) / 255.0

# -------------------------------
# STEP 7: Save Processed Face
# -------------------------------
output_path = os.path.join(OUTPUT_DIR, "processed_face_256.png")

# Convert back to uint8 for saving
face_to_save = (face_array * 255).astype(np.uint8)
Image.fromarray(face_to_save).save(output_path)

print("✅ Face preprocessing completed successfully")
print("📁 Saved at:", output_path)
print("📐 Shape:", face_array.shape)
