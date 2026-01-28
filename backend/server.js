const express = require("express");
const cors = require("cors");

const enrollRoutes = require("./routes/enrollRoutes");
const authRoutes = require("./routes/authRoutes");
const revokeRoutes = require("./routes/revokeRoutes");

const app = express();
const PORT = 5000;

// -----------------------------
// MIDDLEWARE
// -----------------------------
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// -----------------------------
// ROUTES
// -----------------------------
app.use("/api/enroll", enrollRoutes);
app.use("/api/authenticate", authRoutes);
app.use("/api/revoke", revokeRoutes);

// -----------------------------
// HEALTH CHECK
// -----------------------------
app.get("/", (req, res) => {
  res.send("Backend server is running");
});

// -----------------------------
// START SERVER
// -----------------------------
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

// // -----------------------------
// // MIDDLEWARE
// // -----------------------------
// app.use(cors());
// app.use(express.json());                 // REQUIRED
// app.use(express.urlencoded({ extended: true }));

// // -----------------------------
// // ROUTES
// // -----------------------------
// app.use("/api/enroll", enrollRoutes);
// app.use("/api/authenticate", authRoutes);
// app.use("/api/revoke", revokeRoutes);    // ✅ REVOCATION WORKS HERE

// // -----------------------------
// // HEALTH CHECK
// // -----------------------------
// app.get("/", (req, res) => {
//   res.send("Backend server is running");
// });

// // -----------------------------
// // START SERVER
// // -----------------------------
// app.listen(PORT, () => {
//   console.log(`✅ Backend server running on http://localhost:${PORT}`);
// });
