const { spawn } = require("child_process");
const path = require("path");

exports.enrollUser = (req, res) => {
  console.log("📥 Enrollment request received");

  // -----------------------------
  // DEBUG LOGS
  // -----------------------------
  console.log("📦 req.body:", req.body);
  console.log("📁 req.files:", req.files);

  // -----------------------------
  // SAFELY READ USERNAME (FIXED)
  // -----------------------------
  let username = req.body && req.body.username;

  // Handle array case (multer quirk)
  if (Array.isArray(username)) {
    username = username[0];
  }

  console.log("➡️ Username:", username);

  if (!username) {
    console.error("❌ username missing in request body");
    return res.status(400).json({
      message: "Username is required"
    });
  }

  // -----------------------------
  // FILE VALIDATION
  // -----------------------------
  if (!req.files || !req.files.face_image || !req.files.cover_image) {
    console.error("❌ Face image or cover image missing");
    return res.status(400).json({
      message: "face_image and cover_image are required"
    });
  }

  // -----------------------------
  // RESOLVE FILE PATHS
  // -----------------------------
  const faceImagePath = path.resolve(req.files.face_image[0].path);
  const coverImagePath = path.resolve(req.files.cover_image[0].path);

  console.log("🖼️ Face image path:", faceImagePath);
  console.log("🖼️ Cover image path:", coverImagePath);

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
      message: "Enrollment successful",
      python_output: stdoutData.trim()
    });
  });

  python.on("error", (err) => {
    return res.status(500).json({
      message: "Failed to start Python process",
      error: err.message
    });
  });
};
