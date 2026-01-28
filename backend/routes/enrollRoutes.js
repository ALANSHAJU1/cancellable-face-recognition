const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { enrollUser } = require("../controllers/enrollController");

/* -----------------------------
   ENSURE DIRECTORIES EXIST
----------------------------- */
const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const facesDir = path.join(__dirname, "..", "uploads", "faces");
const coversDir = path.join(__dirname, "..", "uploads", "covers");

ensureDir(facesDir);
ensureDir(coversDir);

/* -----------------------------
   MULTER CONFIGURATION
----------------------------- */
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    if (file.fieldname === "face_image") {
      cb(null, facesDir);
    } else if (file.fieldname === "cover_image") {
      cb(null, coversDir);
    }
  },
  filename: (req, file, cb) => {
    const safeName =
      Date.now() +
      "-" +
      file.originalname.replace(/\s+/g, "_");
    cb(null, safeName);
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
   ENROLL API  ✅ FIXED
   POST /api/enroll
----------------------------- */
router.post(
  "/",
  upload.fields([
    { name: "face_image", maxCount: 1 },
    { name: "cover_image", maxCount: 1 }
  ]),
  enrollUser
);

module.exports = router;



// const express = require("express");
// const router = express.Router();
// const multer = require("multer");
// const path = require("path");
// const { enrollUser } = require("../controllers/enrollController");

// /* -----------------------------
//    MULTER CONFIGURATION
// ----------------------------- */

// const storage = multer.diskStorage({
//   destination: (req, file, cb) => {
//     if (file.fieldname === "face_image") {
//       cb(null, "uploads/faces/");
//     } else if (file.fieldname === "cover_image") {
//       cb(null, "uploads/covers/");
//     }
//   },
//   filename: (req, file, cb) => {
//     const uniqueName = Date.now() + "-" + file.originalname;
//     cb(null, uniqueName);
//   }
// });

// const fileFilter = (req, file, cb) => {
//   if (file.mimetype.startsWith("image/")) {
//     cb(null, true);
//   } else {
//     cb(new Error("Only image files allowed"), false);
//   }
// };

// const upload = multer({ storage, fileFilter });

// /* -----------------------------
//    ENROLL API
// ----------------------------- */

// router.post(
//   "/enroll",
//   upload.fields([
//     { name: "face_image", maxCount: 1 },
//     { name: "cover_image", maxCount: 1 }
//   ]),
//   enrollUser
// );

// module.exports = router;
