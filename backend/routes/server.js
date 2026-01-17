const express = require("express");
const app = express();

app.use(express.json());

// ENROLL API
app.use("/api", require("./routes/enrollRoutes"));

// AUTHENTICATE API
app.use("/api", require("./routes/authRoutes"));

app.listen(5000, () => {
  console.log("✅ Server running on port 5000");
});
