// const express = require("express");
// const router = express.Router();
// const multer = require("multer");
// const { authenticateUser } = require("../controllers/authController");

// // Multer configuration for authentication face images
// const upload = multer({
//   dest: "uploads/auth_faces/"
// });

// // POST /api/authenticate
// // Accepts:
// // - username (text)
// // - face_image (file)
// router.post(
//   "/authenticate",
//   upload.single("face_image"),
//   authenticateUser
// );

// module.exports = router;


const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { authenticateUser } = require("../controllers/authController");

// -----------------------------
// Ensure upload dir
// -----------------------------
const uploadDir = path.join(__dirname, "..", "uploads", "auth_faces");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// -----------------------------
// Multer config (AUTH)
// -----------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const safeName =
      Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
    cb(null, safeName);
  }
});

const upload = multer({ storage });

// -----------------------------
// AUTHENTICATION API
// -----------------------------
router.post(
  "/",
  upload.single("face_image"),   // 🔥 THIS WAS MISSING
  authenticateUser
);

module.exports = router;
