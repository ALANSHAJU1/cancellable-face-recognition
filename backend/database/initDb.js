const sqlite3 = require("sqlite3").verbose();
const path = require("path");

// Path to SQLite DB
const dbPath = path.join(__dirname, "app.db");

// Connect (creates DB if it doesn't exist)
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error("❌ Failed to connect to SQLite:", err.message);
  } else {
    console.log("✅ Connected to SQLite database");
  }
});

// Create templates table
const createTableQuery = `
CREATE TABLE IF NOT EXISTS templates (
    template_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    stego_image_encrypted BLOB NOT NULL,
    encryption_iv BLOB NOT NULL,
    encryption_tag BLOB NOT NULL,
    feature_extractor TEXT,
    cover_image_hash TEXT,
    quality_psnr REAL,
    quality_ssim REAL,
    status TEXT DEFAULT 'ACTIVE',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
`;

db.run(createTableQuery, (err) => {
  if (err) {
    console.error("❌ Failed to create table:", err.message);
  } else {
    console.log("✅ Templates table created (or already exists)");
  }
});

// Close DB
db.close((err) => {
  if (err) {
    console.error("❌ Failed to close DB:", err.message);
  } else {
    console.log("✅ SQLite DB initialization complete");
  }
});
