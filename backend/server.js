const express = require("express");
const cors = require("cors");

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());


// Routes
app.use("/api", require("./routes/enrollRoutes"));
app.use("/api", require("./routes/authRoutes"));

// Health check (VERY IMPORTANT)
app.get("/", (req, res) => {
  res.send("Backend server is running");
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Backend server running on http://localhost:${PORT}`);
});
