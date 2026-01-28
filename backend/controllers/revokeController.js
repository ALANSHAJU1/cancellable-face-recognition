// const sqlite3 = require("sqlite3").verbose();
// const path = require("path");
// const fs = require("fs");

// // -------------------------------
// // DATABASE PATH (ABSOLUTE)
// // -------------------------------
// const DB_PATH = path.join(
//   __dirname,
//   "..",
//   "database",
//   "app.db"
// );

// // -------------------------------
// // ML KEY STORAGE PATH
// // -------------------------------
// const KEYS_DIR = path.join(
//   __dirname,
//   "..",
//   "..",
//   "ml_backend",
//   "keys"
// );

// // -----------------------------------
// // POST /api/revoke
// // -----------------------------------
// exports.revokeTemplate = (req, res) => {
//   console.log("🔥 Revoke controller hit");

//   const { username } = req.body;

//   if (!username) {
//     return res.status(400).json({
//       success: false,
//       message: "username is required",
//     });
//   }

//   const normalizedUser = username.trim().toLowerCase();

//   const db = new sqlite3.Database(DB_PATH, sqlite3.OPEN_READWRITE, (err) => {
//     if (err) {
//       console.error("❌ DB open error:", err);
//       return res.status(500).json({
//         success: false,
//         message: "Database connection failed",
//       });
//     }
//   });

//   db.run(
//     `
//     UPDATE templates
//     SET status = 'REVOKED'
//     WHERE user_id = ? AND status = 'ACTIVE'
//     `,
//     [normalizedUser],
//     function (err) {
//       if (err) {
//         console.error("❌ Revocation DB error:", err);
//         db.close();
//         return res.status(500).json({
//           success: false,
//           message: "Revocation failed",
//         });
//       }

//       if (this.changes === 0) {
//         console.warn(`⚠️ No active template for user: ${normalizedUser}`);
//         db.close();
//         return res.status(404).json({
//           success: false,
//           message: "No active template found to revoke",
//         });
//       }

//       // -------------------------------
//       // 🔥 HARD REVOCATION (Design B)
//       // -------------------------------
//       const keyPath = path.join(KEYS_DIR, `${normalizedUser}.key`);
//       const rPath = path.join(KEYS_DIR, `${normalizedUser}_R.npy`);

//       try {
//         if (fs.existsSync(keyPath)) {
//           fs.unlinkSync(keyPath);
//         }
//         if (fs.existsSync(rPath)) {
//           fs.unlinkSync(rPath);
//         }

//         console.log(`🔐 Keys deleted for user: ${normalizedUser}`);
//       } catch (fileErr) {
//         // DO NOT FAIL — DB revocation already succeeded
//         console.warn("⚠️ Key deletion warning:", fileErr.message);
//       }

//       db.close();

//       return res.json({
//         success: true,
//         message: "Template revoked successfully",
//       });
//     }
//   );
// };




const sqlite3 = require("sqlite3").verbose();
const path = require("path");
const fs = require("fs");

// -------------------------------
// DATABASE PATH (ABSOLUTE)
// -------------------------------
const DB_PATH = path.join(
  __dirname,
  "..",
  "database",
  "app.db"
);

// -------------------------------
// ML KEY STORAGE PATH
// -------------------------------
const KEYS_DIR = path.join(
  __dirname,
  "..",
  "..",
  "ml_backend",
  "keys"
);

// -----------------------------------
// POST /api/revoke
// -----------------------------------
exports.revokeTemplate = (req, res) => {
  console.log("🔥 Revoke controller hit");

  const { username } = req.body;

  if (!username) {
    return res.status(400).json({
      success: false,
      message: "username is required",
    });
  }

  const normalizedUser = username.trim().toLowerCase();

  const db = new sqlite3.Database(DB_PATH, sqlite3.OPEN_READWRITE, (err) => {
    if (err) {
      console.error("❌ DB open error:", err);
      return res.status(500).json({
        success: false,
        message: "Database connection failed",
      });
    }
  });

  db.run(
    `
    UPDATE templates
    SET status = 'REVOKED'
    WHERE user_id = ? AND status = 'ACTIVE'
    `,
    [normalizedUser],
    function (err) {
      if (err) {
        console.error("❌ Revocation DB error:", err);
        db.close();
        return res.status(500).json({
          success: false,
          message: "Revocation failed",
        });
      }

      if (this.changes === 0) {
        db.close();
        return res.status(404).json({
          success: false,
          message: "No active template found to revoke",
        });
      }

      // 🔥 HARD REVOCATION — delete keys
      const keyPath = path.join(KEYS_DIR, `${normalizedUser}.key`);
      const rPath = path.join(KEYS_DIR, `${normalizedUser}_R.npy`);

      try {
        if (fs.existsSync(keyPath)) fs.unlinkSync(keyPath);
        if (fs.existsSync(rPath)) fs.unlinkSync(rPath);
        console.log(`🔐 Keys deleted for user: ${normalizedUser}`);
      } catch (err) {
        console.warn("⚠️ Key deletion warning:", err.message);
      }

      db.close();

      return res.json({
        success: true,
        message: "Template revoked successfully",
      });
    }
  );
};

// -----------------------------------
// ✅ GET /api/revoke/status/:username
// (REGENERATION GUARD)
// -----------------------------------
exports.checkRevocationStatus = (req, res) => {
  const username = req.params.username;

  if (!username) {
    return res.json({ allowed: false });
  }

  const normalizedUser = username.trim().toLowerCase();
  const db = new sqlite3.Database(DB_PATH);

  db.get(
    `
    SELECT status
    FROM templates
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT 1
    `,
    [normalizedUser],
    (err, row) => {
      db.close();

      if (err || !row) {
        return res.json({ allowed: false });
      }

      return res.json({
        allowed: row.status === "REVOKED",
      });
    }
  );
};
