const express = require("express");
const router = express.Router();
const multer = require("multer");
const { authenticateUser } = require("../controllers/authController");

// Multer configuration for authentication face images
const upload = multer({
  dest: "uploads/auth_faces/"
});

// POST /api/authenticate
// Accepts:
// - username (text)
// - face_image (file)
router.post(
  "/authenticate",
  upload.single("face_image"),
  authenticateUser
);

module.exports = router;
