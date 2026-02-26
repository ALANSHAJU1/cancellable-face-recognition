//authController.js
const { spawn } = require("child_process");
const path = require("path");

exports.authenticateUser = (req, res) => {
  console.log("Authentication request received");

  if (!req.body?.username) {
    return res.status(400).json({
      decision: "REJECT",
      reason: "username is required"
    });
  }

  if (!req.file) {
    return res.status(400).json({
      decision: "REJECT",
      reason: "face_image is required"
    });
  }

  //  NORMALIZE (MUST MATCH ENROLL + PYTHON)
  const username = req.body.username
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");

  const faceImagePath = path.resolve(req.file.path);

  console.log(" Normalized Username:", username);
  console.log(" Face image path:", faceImagePath);

  const pythonScriptPath = path.resolve(
    __dirname,
    "../../ml_backend/inference/authenticate.py"
  );

  console.log(" Python script:", pythonScriptPath);

  const python = spawn("python", [
    pythonScriptPath,
    username,
    faceImagePath
  ]);

  let stdout = "";
  let stderr = "";

  python.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    console.log(" PYTHON STDOUT:", text);
  });

  python.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    console.error(" PYTHON STDERR:", text);
  });

  python.on("close", () => {
    if (!stdout) {
      console.error(" Empty Python output");
      return res.status(500).json({
        decision: "REJECT",
        reason: "No response from authentication engine"
      });
    }

    try {
      const result = JSON.parse(stdout);
      return res.json(result);
    } catch (err) {
      console.error(" JSON parse error:", err.message);
      return res.status(500).json({
        decision: "REJECT",
        reason: "Invalid authentication response"
      });
    }
  });

  python.on("error", (err) => {
    console.error("Python spawn failed:", err.message);
    return res.status(500).json({
      decision: "REJECT",
      reason: "Authentication engine launch failed"
    });
  });
};

