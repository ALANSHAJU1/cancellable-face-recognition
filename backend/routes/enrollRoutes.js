const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { enrollUser } = require("../controllers/enrollController");

/* -----------------------------
   MULTER CONFIGURATION
----------------------------- */

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    if (file.fieldname === "face_image") {
      cb(null, "uploads/faces/");
    } else if (file.fieldname === "cover_image") {
      cb(null, "uploads/covers/");
    }
  },
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + "-" + file.originalname;
    cb(null, uniqueName);
  }
});

const fileFilter = (req, file, cb) => {
  if (file.mimetype.startsWith("image/")) {
    cb(null, true);
  } else {
    cb(new Error("Only image files allowed"), false);
  }
};

const upload = multer({ storage, fileFilter });

/* -----------------------------
   ENROLL API
----------------------------- */

router.post(
  "/enroll",
  upload.fields([
    { name: "face_image", maxCount: 1 },
    { name: "cover_image", maxCount: 1 }
  ]),
  enrollUser
);

module.exports = router;
