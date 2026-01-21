const { spawn } = require("child_process");
const path = require("path");

exports.authenticateUser = (req, res) => {
  console.log("🔐 Authentication request received");

  // Validate username
  if (!req.body || !req.body.username) {
    return res.status(400).json({
      message: "username is required"
    });
  }

  // Validate face image
  if (!req.file) {
    return res.status(400).json({
      message: "face_image is required"
    });
  }

  // ✅ CRITICAL FIX: normalize username (MUST MATCH ENROLL + PYTHON)
  const username = req.body.username.trim().toLowerCase();
  const faceImagePath = path.resolve(req.file.path);

  console.log("➡️ Username:", username);
  console.log("🖼️ Face image path:", faceImagePath);

  // Absolute path to Python script
  const pythonScriptPath = path.join(
    __dirname,
    "..",
    "..",
    "ml_backend",
    "inference",
    "authenticate.py"
  );

  console.log("🐍 Python script:", pythonScriptPath);

  const python = spawn("python", [
    pythonScriptPath,
    username,
    faceImagePath
  ]);

  let output = "";
  let errorOutput = "";

  python.stdout.on("data", (data) => {
    output += data.toString();
  });

  python.stderr.on("data", (data) => {
    errorOutput += data.toString();
  });

  python.on("close", (code) => {
    if (code !== 0) {
      console.error("❌ Python error:", errorOutput);
      return res.status(500).json({
        message: "Authentication failed",
        error: errorOutput
      });
    }

    try {
      const result = JSON.parse(output);
      return res.json(result);
    } catch {
      return res.json({
        message: output.trim()
      });
    }
  });
};
