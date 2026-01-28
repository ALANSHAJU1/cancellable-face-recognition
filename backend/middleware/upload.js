// const multer = require("multer");
// const fs = require("fs");
// const path = require("path");

// function ensureDir(dir) {
//   if (!fs.existsSync(dir)) {
//     fs.mkdirSync(dir, { recursive: true });
//   }
// }

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

//   filename: function (req, file, cb) {
//     cb(null, Date.now() + "-" + file.originalname);
//   }
// });

// module.exports = multer({ storage });


const multer = require("multer");
const fs = require("fs");
const path = require("path");

// -----------------------------
// Ensure directory exists
// -----------------------------
function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// -----------------------------
// Multer Storage Configuration
// -----------------------------
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    let uploadDir = "";

    if (file.fieldname === "face_image") {
      uploadDir = path.join(__dirname, "..", "uploads", "faces");
    } else if (file.fieldname === "cover_image") {
      uploadDir = path.join(__dirname, "..", "uploads", "covers");
    }

    ensureDir(uploadDir);
    cb(null, uploadDir);
  },

  // 🔥 FIX 3: SANITIZE FILENAME (IMPORTANT)
  filename: function (req, file, cb) {
    const safeName =
      Date.now() + "-" + file.originalname.replace(/\s+/g, "_");
    cb(null, safeName);
  }
});

module.exports = multer({ storage });
