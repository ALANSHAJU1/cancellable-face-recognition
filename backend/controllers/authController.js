const { spawn } = require("child_process");

exports.authenticateUser = (req, res) => {
  const { user_id } = req.body;

  if (!user_id) {
    return res.status(400).json({ error: "user_id is required" });
  }

  const python = spawn("python", [
    "ml_backend/inference/authenticate.py",
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
        message: "Authentication failed",
        error: errorOutput
      });
    }

    try {
      const result = JSON.parse(output);
      return res.json(result);
    } catch {
      return res.json({ message: output.trim() });
    }
  });
};









// const { spawn } = require("child_process");

// exports.authenticateUser = (req, res) => {
//   const userId = req.body.user_id;

//   if (!userId) {
//     return res.status(400).json({ error: "User ID required" });
//   }

//   const python = spawn("python", [
//     "ml_backend/inference/authenticate.py",
//     userId
//   ]);

//   let output = "";

//   python.stdout.on("data", (data) => {
//     output += data.toString();
//   });

//   python.stderr.on("data", (data) => {
//     console.error("Python error:", data.toString());
//   });

//   python.on("close", () => {
//     res.json({
//       message: "Authentication completed",
//       result: output
//     });
//   });
// };
