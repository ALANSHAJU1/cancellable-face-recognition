//backend/middleware/upload.js

const multer = require("multer");
const fs = require("fs");
const path = require("path");

// Ensure directory exists
function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}
// Base Upload Folder
const baseUploadDir = path.join(__dirname, "..", "uploads");

// ENROLLMENT STORAGE
const enrollStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    let uploadDir = "";

    if (file.fieldname === "face_image") {
      uploadDir = path.join(baseUploadDir, "faces");
    } else if (file.fieldname === "cover_image") {
      uploadDir = path.join(baseUploadDir, "covers");
    }

    ensureDir(uploadDir);
    cb(null, uploadDir);
  },

  filename: function (req, file, cb) {
    const safeName =
      Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
    cb(null, safeName);
  }
});

// AUTHENTICATION STORAGE
const authStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(baseUploadDir, "auth_faces");
    ensureDir(uploadDir);
    cb(null, uploadDir);
  },

  filename: function (req, file, cb) {
    const safeName =
      Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
    cb(null, safeName);
  }
});

// File Filter (Optional Security)
function fileFilter(req, file, cb) {
  if (!file.mimetype.startsWith("image/")) {
    return cb(new Error("Only image files are allowed"), false);
  }
  cb(null, true);
}

// Export Middlewares
const enrollUpload = multer({
  storage: enrollStorage,
  fileFilter,
  //limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

const authUpload = multer({
  storage: authStorage,
  fileFilter,
  //limits: { fileSize: 5 * 1024 * 1024 }
});

module.exports = {
  enrollUpload,
  authUpload
};































// //backend/middleware/upload.js

// const multer = require("multer");
// const fs = require("fs");
// const path = require("path");

// // Ensure directory exists
// function ensureDir(dir) {
//   if (!fs.existsSync(dir)) {
//     fs.mkdirSync(dir, { recursive: true });
//   }
// }

// // Multer Storage Configuration
// const storage = multer.diskStorage({
//   destination: function (req, file, cb) {
//     let uploadDir = "";

//     if (file.fieldname === "face_image") {
//       uploadDir = path.join(__dirname, "..", "uploads", "faces");
//     } else if (file.fieldname === "cover_image") {
//       uploadDir = path.join(__dirname, "..", "uploads", "covers");
//     }

//     ensureDir(uploadDir);
//     cb(null, uploadDir);
//   },

//   // SANITIZE FILENAME TO PREVENT ISSUES
//   filename: function (req, file, cb) {
//     const safeName =
//       Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
//     cb(null, safeName);
//   }
// });

// module.exports = multer({ storage });














