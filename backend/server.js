// backend/server.js
const express = require("express");
const cors = require("cors");
const multer = require("multer");
//require("dotenv").config(); // For environment variables 

// Route Imports
const enrollRoutes = require("./routes/enrollRoutes");
const authRoutes = require("./routes/authRoutes");
const revokeRoutes = require("./routes/revokeRoutes");

const app = express();
const PORT = process.env.PORT || 5000;

// Enable CORS (You can restrict origin in production)
app.use(
  cors({
    origin: "http://localhost:3000", // Change in production
    methods: ["GET", "POST", "PUT", "DELETE"],
    credentials: true
  })
);

// Parse JSON body
app.use(express.json());

// Parse URL-encoded data
app.use(express.urlencoded({ extended: true }));

// ROUTES
app.use("/api/enroll", enrollRoutes);
app.use("/api/authenticate", authRoutes);
app.use("/api/revoke", revokeRoutes);

// HEALTH CHECK
app.get("/", (req, res) => {
  res.status(200).json({
    status: "OK",
    message: "Backend server is running"
  });
});

// 404 HANDLER (Route Not Found)
app.use((req, res) => {
  res.status(404).json({
    error: "Route not found"
  });
});

// GLOBAL ERROR HANDLER
app.use((err, req, res, next) => {
  console.error(" Global Error Handler:", err.message);

  // Multer errors (file upload issues)
  if (err instanceof multer.MulterError) {
    return res.status(400).json({
      error: "File Upload Error",
      message: err.message
    });
  }

  // Custom errors
  if (err) {
    return res.status(500).json({
      error: "Internal Server Error",
      message: err.message
    });
  }

  next();
});

// START SERVER
app.listen(PORT, () => {
  console.log(`✅ Backend server running on http://localhost:${PORT}`);
});




















// const express = require("express");
// const cors = require("cors");

// const enrollRoutes = require("./routes/enrollRoutes");
// const authRoutes = require("./routes/authRoutes");
// const revokeRoutes = require("./routes/revokeRoutes");

// const app = express();
// const PORT = 5000;

// // MIDDLEWARE
// app.use(cors());
// app.use(express.json());
// app.use(express.urlencoded({ extended: true }));

// // ROUTES
// app.use("/api/enroll", enrollRoutes);
// app.use("/api/authenticate", authRoutes);
// app.use("/api/revoke", revokeRoutes);

// // HEALTH CHECK
// app.get("/", (req, res) => {
//   res.send("Backend server is running");
// });

// // START SERVER
// app.listen(PORT, () => {
//   console.log(`Backend server running on http://localhost:${PORT}`);
// });








