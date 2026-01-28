// const { spawn } = require("child_process");
// const path = require("path");

// exports.enrollUser = (req, res) => {
//   console.log("📥 Enrollment request received");

//   // -----------------------------
//   // DEBUG LOGS
//   // -----------------------------
//   console.log("📦 req.body:", req.body);
//   console.log("📁 req.files:", req.files);

//   // -----------------------------
//   // READ + NORMALIZE USERNAME (CRITICAL FIX)
//   // -----------------------------
//   let username = req.body && req.body.username;

//   if (Array.isArray(username)) {
//     username = username[0];
//   }

//   if (!username) {
//     console.error("❌ username missing");
//     return res.status(400).json({ message: "Username is required" });
//   }

//   // ✅ NORMALIZATION (MUST MATCH PYTHON)
//   username = username.trim().toLowerCase();

//   console.log("➡️ Normalized Username:", username);

//   // -----------------------------
//   // FILE VALIDATION
//   // -----------------------------
//   if (!req.files || !req.files.face_image || !req.files.cover_image) {
//     console.error("❌ Face or cover image missing");
//     return res.status(400).json({
//       message: "face_image and cover_image are required"
//     });
//   }

//   // -----------------------------
//   // RESOLVE FILE PATHS
//   // -----------------------------
//   const faceImagePath = path.resolve(req.files.face_image[0].path);
//   const coverImagePath = path.resolve(req.files.cover_image[0].path);

//   console.log("🖼️ Face image path:", faceImagePath);
//   console.log("🖼️ Cover image path:", coverImagePath);

//   // -----------------------------
//   // PYTHON SCRIPT PATH (ALREADY FIXED)
//   // -----------------------------
//   const pythonScriptPath = path.resolve(
//     __dirname,
//     "../../ml_backend/inference/enroll.py"
//   );

//   console.log("🐍 Python script:", pythonScriptPath);

//   // -----------------------------
//   // SPAWN PYTHON
//   // -----------------------------
//   const python = spawn("python", [
//     pythonScriptPath,
//     username,
//     faceImagePath,
//     coverImagePath
//   ]);

//   let stdoutData = "";
//   let stderrData = "";

//   python.stdout.on("data", (data) => {
//     stdoutData += data.toString();
//     console.log("🐍 Python STDOUT:", data.toString());
//   });

//   python.stderr.on("data", (data) => {
//     stderrData += data.toString();
//     console.error("❌ Python STDERR:", data.toString());
//   });

//   python.on("close", (code) => {
//     console.log("🐍 Python exited with code:", code);

//     if (code !== 0) {
//       return res.status(500).json({
//         message: "Enrollment failed",
//         python_error: stderrData || "Unknown Python error"
//       });
//     }

//     return res.json({
//       message: "Enrollment successful",
//       user_id: username
//     });
//   });

//   python.on("error", (err) => {
//     return res.status(500).json({
//       message: "Failed to start Python process",
//       error: err.message
//     });
//   });
// };


const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const sqlite3 = require("sqlite3").verbose();

const DB_PATH = path.join(__dirname, "..", "database", "app.db");

exports.enrollUser = (req, res) => {
  // 🔥🔥🔥 CRITICAL DEBUG LINE (DO NOT REMOVE)
  console.log("🔥🔥🔥 ENROLL CONTROLLER HIT 🔥🔥🔥");

  console.log("📦 req.body:", req.body);
  console.log("📁 req.files:", req.files);

  // -----------------------------
  // READ + NORMALIZE USERNAME
  // -----------------------------
  let username = req.body?.username;

  if (Array.isArray(username)) {
    username = username[0];
  }

  if (!username) {
    console.error("❌ Username missing");
    return res.status(400).json({ message: "Username is required" });
  }

  // ✅ MUST MATCH PYTHON EXACTLY
  username = username.trim().toLowerCase().replace(/\s+/g, "_");

  console.log("➡️ Normalized Username:", username);

  // -----------------------------
  // FILE VALIDATION
  // -----------------------------
  if (!req.files?.face_image || !req.files?.cover_image) {
    console.error("❌ Face or cover image missing");
    return res.status(400).json({
      message: "face_image and cover_image are required"
    });
  }

  const faceImagePath = path.resolve(req.files.face_image[0].path);
  const coverImagePath = path.resolve(req.files.cover_image[0].path);

  // 🔍 ABSOLUTE PATH VERIFICATION (CRITICAL)
  console.log("📂 Face path:", faceImagePath);
  console.log("📂 Cover path:", coverImagePath);
  console.log("✅ Face exists:", fs.existsSync(faceImagePath));
  console.log("✅ Cover exists:", fs.existsSync(coverImagePath));

  if (!fs.existsSync(faceImagePath) || !fs.existsSync(coverImagePath)) {
    return res.status(500).json({
      message: "Uploaded files not found on server"
    });
  }

  // -----------------------------
  // ENFORCE ONE-ACTIVE-TEMPLATE RULE
  // -----------------------------
  const db = new sqlite3.Database(DB_PATH);
  db.run(
    `UPDATE templates SET status='REVOKED' WHERE user_id=? AND status='ACTIVE'`,
    [username],
    () => db.close()
  );

  // -----------------------------
  // PYTHON SCRIPT PATH
  // -----------------------------
  const pythonScriptPath = path.resolve(
    __dirname,
    "../../ml_backend/inference/enroll.py"
  );

  console.log("🐍 Python script:", pythonScriptPath);

  // -----------------------------
  // SPAWN PYTHON
  // -----------------------------
  const python = spawn("python", [
    pythonScriptPath,
    username,
    faceImagePath,
    coverImagePath
  ]);

  let stdoutData = "";
  let stderrData = "";

  python.stdout.on("data", (data) => {
    stdoutData += data.toString();
    console.log("🐍 Python STDOUT:", data.toString());
  });

  python.stderr.on("data", (data) => {
    stderrData += data.toString();
    console.error("❌ Python STDERR:", data.toString());
  });

  python.on("close", (code) => {
    console.log("🐍 Python exited with code:", code);

    if (code !== 0) {
      return res.status(500).json({
        message: "Enrollment failed",
        python_error: stderrData || "Unknown Python error"
      });
    }

    return res.json({
      success: true,
      message: "Enrollment successful",
      user_id: username
    });
  });

  python.on("error", (err) => {
    console.error("❌ Failed to start Python:", err.message);
    return res.status(500).json({
      message: "Failed to start Python process",
      error: err.message
    });
  });
};
