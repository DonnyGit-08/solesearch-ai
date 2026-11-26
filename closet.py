"""Digital closet stored in a local SQLite database (one file, no server needed)."""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key TEXT NOT NULL,
    colorway_key TEXT,
    size TEXT,
    condition TEXT,
    purchase_price REAL,
    purchased_on TEXT,
    notes TEXT,
    photo_path TEXT,
    wear_count INTEGER DEFAULT 0,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_path TEXT,
    predicted_model TEXT,
    confirmed_model TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);
"""


class Closet:
    def __init__(self, db_path: Path, photo_dir: Path):
        self.photo_dir = photo_dir
        photo_dir.mkdir(parents=True, exist_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save_photo(self, img: Image.Image) -> str:
        img = img.convert("RGB")
        img.thumbnail((800, 800))
        path = self.photo_dir / f"{uuid.uuid4().hex}.jpg"
        img.save(path, "JPEG", quality=88)
        return str(path)

    def add_pair(self, model_key, colorway_key, size, condition, purchase_price,
                 purchased_on, notes, photo_path):
        cur = self.conn.execute(
            """INSERT INTO pairs (model_key, colorway_key, size, condition, purchase_price,
                                  purchased_on, notes, photo_path, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_key, colorway_key, size, condition, purchase_price, purchased_on,
             notes, photo_path, datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()
        return cur.lastrowid

    def log_correction(self, photo_path, predicted_model, confirmed_model, confidence):
        """Called when the user picks a different model than the AI's first guess.
        These photos and correct labels can be added to train_data to improve the model."""
        self.conn.execute(
            """INSERT INTO corrections (photo_path, predicted_model, confirmed_model,
                                        confidence, created_at) VALUES (?, ?, ?, ?, ?)""",
            (photo_path, predicted_model, confirmed_model, confidence,
             datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()

    def pairs(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM pairs ORDER BY added_at DESC")]

    def corrections(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM corrections ORDER BY created_at DESC")]

    def add_wear(self, pair_id):
        self.conn.execute("UPDATE pairs SET wear_count = wear_count + 1 WHERE id = ?", (pair_id,))
        self.conn.commit()

    def delete_pair(self, pair_id):
        row = self.conn.execute("SELECT photo_path FROM pairs WHERE id = ?", (pair_id,)).fetchone()
        self.conn.execute("DELETE FROM pairs WHERE id = ?", (pair_id,))
        self.conn.commit()
        if row and row["photo_path"]:
            # keep photos that the corrections log still points to
            used = self.conn.execute("SELECT 1 FROM corrections WHERE photo_path = ?",
                                     (row["photo_path"],)).fetchone()
            if not used:
                Path(row["photo_path"]).unlink(missing_ok=True)
