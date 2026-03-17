"""
One-time migration: user_training_data.json + scraped_articles.json → airia.db
Run this ONCE before switching to the new main.py.
Safe to re-run — skips articles that already exist, and checks if training data was already migrated.
"""

import json
import sqlite3
from pathlib import Path

DB_FILE = Path("airia.db")
TRAINING_DATA_FILE = Path("user_training_data.json")
ARTICLES_FILE = Path("scraped_articles.json")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS training_data (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            avg_wpm          REAL,
            wpm_variance     REAL,
            back_presses     REAL,
            completion_rate  REAL,
            slowdown_ratio   REAL,
            blur_count       REAL,
            label            INTEGER,
            timestamp        TEXT
        );

        CREATE TABLE IF NOT EXISTS articles (
            id               TEXT PRIMARY KEY,
            url              TEXT,
            title            TEXT,
            content          TEXT,
            estimated_lexile INTEGER,
            word_count       INTEGER,
            paragraph_count  INTEGER,
            broad_genre      TEXT,
            specific_genre   TEXT,
            genre_difficulty REAL,
            genre_reasoning  TEXT,
            timestamp        TEXT
        );
    """)
    conn.commit()


def migrate_training_data(conn):
    if not TRAINING_DATA_FILE.exists():
        print("No user_training_data.json found — skipping training data migration.")
        return

    with open(TRAINING_DATA_FILE, "r") as f:
        data = json.load(f)

    if not data:
        print("user_training_data.json is empty — nothing to migrate.")
        return

    existing = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
    if existing > 0:
        print(f"training_data table already has {existing} rows — skipping to avoid duplicates.")
        print("If you want to re-migrate, manually run: DELETE FROM training_data;")
        return

    inserted = 0
    for sample in data:
        features = sample["features"]
        conn.execute(
            """INSERT INTO training_data
               (avg_wpm, wpm_variance, back_presses, completion_rate,
                slowdown_ratio, blur_count, label, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (*features, sample["label"], sample.get("timestamp", ""))
        )
        inserted += 1

    conn.commit()
    print(f"Migrated {inserted} training samples from user_training_data.json")


def migrate_articles(conn):
    if not ARTICLES_FILE.exists():
        print("No scraped_articles.json found — skipping articles migration.")
        return

    with open(ARTICLES_FILE, "r") as f:
        articles = json.load(f)

    if not articles:
        print("scraped_articles.json is empty — nothing to migrate.")
        return

    inserted = 0
    skipped = 0

    for article in articles:
        c = article.get("classification", {})
        try:
            conn.execute(
                """INSERT INTO articles
                   (id, url, title, content, estimated_lexile, word_count,
                    paragraph_count, broad_genre, specific_genre,
                    genre_difficulty, genre_reasoning, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article["id"], article.get("url", ""), article.get("title", ""),
                    article.get("content", ""), article.get("estimated_lexile", 0),
                    article.get("word_count", 0), article.get("paragraph_count", 0),
                    c.get("broad_genre", "other"), c.get("specific_genre", "unknown"),
                    c.get("genre_difficulty", 0.5), c.get("reasoning", ""),
                    article.get("timestamp", "")
                )
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # Article ID already exists
            skipped += 1

    conn.commit()
    print(f"Migrated {inserted} articles from scraped_articles.json ({skipped} skipped, already existed)")


if __name__ == "__main__":
    print(f"Migrating to {DB_FILE}...")
    conn = get_db()
    init_db(conn)
    migrate_training_data(conn)
    migrate_articles(conn)
    conn.close()
    print("\nMigration complete. You can now run the new main.py.")
    print("Your JSON files are untouched — keep them as backup until you've verified everything works.")