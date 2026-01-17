import cv2
from mtcnn import MTCNN
import os

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = "../datasets/celeba/img_align_celeba/img_align_celeba/sample.jpg"

# -------------------------------
# Load image
# -------------------------------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise ValueError("Image not found")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------
# Pretrained MTCNN
# -------------------------------
detector = MTCNN()
results = detector.detect_faces(img_rgb)

if len(results) == 0:
    raise ValueError("No face detected")

face = results[0]

x, y, w, h = face["box"]
landmarks = face["keypoints"]

# -------------------------------
# Draw bounding box
# -------------------------------
cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

# -------------------------------
# Draw landmarks
# -------------------------------
for key, point in landmarks.items():
    cv2.circle(img_bgr, point, 3, (0, 0, 255), -1)

# -------------------------------
# Show result
# -------------------------------
cv2.imshow("MTCNN Verification", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
