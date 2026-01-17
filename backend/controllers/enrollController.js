const { spawn } = require("child_process");

exports.enrollUser = (req, res) => {
  const { user_id } = req.body;

  if (!user_id) {
    return res.status(400).json({ error: "user_id is required" });
  }

  // Call Python enrollment script
  const python = spawn("python", [
    "ml_backend/inference/enroll.py",
    user_id
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
      return res.status(500).json({
        message: "Enrollment failed",
        error: errorOutput
      });
    }

    return res.json({
      message: "Enrollment successful",
      result: output.trim()
    });
  });
};
