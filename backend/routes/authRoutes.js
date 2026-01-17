const express = require("express");
const router = express.Router();
const { authenticateUser } = require("../controllers/authController");

router.post("/authenticate", authenticateUser);

module.exports = router;
