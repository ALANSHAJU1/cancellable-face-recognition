// const express = require("express");
// const router = express.Router();

// const { revokeTemplate } = require("../controllers/revokeController");

// // POST /api/revoke
// router.post("/", revokeTemplate);

// module.exports = router;




const express = require("express");
const router = express.Router();

const {
  revokeTemplate,
  checkRevocationStatus
} = require("../controllers/revokeController");

// POST /api/revoke
router.post("/", revokeTemplate);

// ✅ REGENERATION GUARD
// GET /api/revoke/status/:username
router.get("/status/:username", checkRevocationStatus);

module.exports = router;
