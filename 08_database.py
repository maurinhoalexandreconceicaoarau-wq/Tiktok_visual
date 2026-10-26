import sqlite3

import pandas as pd

from common import (
    CLEANED_FILE, DATABASE_FILE, ensure_dirs, read_csv, require,
    setup_console,
)

setup_console()
ensure_dirs()

print("=" * 55)
print("STEP 8 - DATABASE")
print("=" * 55)

# ==========================================================
# 1. LOAD CLEANED DATA
# ==========================================================

require(CLEANED_FILE, "01_cleaning.py")
df = read_csv(CLEANED_FILE)

print(f"\nCleaned data: {CLEANED_FILE}")
print(f"Shape: {df.shape}")

# ==========================================================
# 2. SCHEMA
# ==========================================================
# Two tables: one row per video and one row per comment. Video fields
# (url, caption, label) repeat on every comment in the CSV, so they are
# stored once in `videos` and linked by video_id.
# IDs are stored as TEXT, the same way the CSV files are read.
# True/False columns are stored as 1/0 (SQLite has no boolean type).

SCHEMA = """
CREATE TABLE videos (
    video_id       TEXT PRIMARY KEY,
    video_url      TEXT,
    video_caption  TEXT,
    video_label    TEXT
);

CREATE TABLE comments (
    comment_id           TEXT PRIMARY KEY,
    video_id             TEXT NOT NULL REFERENCES videos (video_id),
    comment              TEXT NOT NULL,
    clean_comment        TEXT NOT NULL,
    likes                INTEGER NOT NULL,
    replies              INTEGER NOT NULL,
    created_at           TEXT,
    date                 TEXT,
    year_month           TEXT,
    language             TEXT,
    username             TEXT,
    is_reply             INTEGER NOT NULL,
    pinned_by_creator    INTEGER NOT NULL,
    liked_by_creator     INTEGER NOT NULL,
    has_text             INTEGER NOT NULL,
    is_repeated_comment  INTEGER NOT NULL,
    comment_length       INTEGER NOT NULL,
    word_count           INTEGER NOT NULL
);

CREATE INDEX idx_comments_video ON comments (video_id);
CREATE INDEX idx_comments_date ON comments (date);
CREATE INDEX idx_comments_username ON comments (username);

-- Same columns as cleaned_comments.csv, for easy querying.
CREATE VIEW cleaned_comments AS
SELECT c.comment_id, c.video_id, v.video_url, v.video_caption, c.comment,
       c.likes, c.replies, c.created_at, c.language, c.is_reply,
       c.username, c.pinned_by_creator, c.liked_by_creator,
       c.clean_comment, c.has_text, c.comment_length, c.word_count,
       c.is_repeated_comment, c.date, c.year_month, v.video_label
FROM comments c
JOIN videos v ON v.video_id = c.video_id;
"""

VIDEO_COLUMNS = ["video_id", "video_url", "video_caption", "video_label"]
COMMENT_COLUMNS = [
    "comment_id", "video_id", "comment", "clean_comment", "likes",
    "replies", "created_at", "date", "year_month", "language", "username",
    "is_reply", "pinned_by_creator", "liked_by_creator", "has_text",
    "is_repeated_comment", "comment_length", "word_count",
]
BOOL_COLUMNS = ["is_reply", "pinned_by_creator", "liked_by_creator",
                "has_text", "is_repeated_comment"]

# ==========================================================
# 3. PREPARE TABLES
# ==========================================================
# Columns that the scraper export did not contain are left empty (text)
# or 0 (flags), so the schema is the same for every dataset.

for col in VIDEO_COLUMNS + COMMENT_COLUMNS:
    if col not in df.columns:
        df[col] = False if col in BOOL_COLUMNS else None

for col in BOOL_COLUMNS:
    df[col] = df[col].astype(bool).astype(int)

videos = df[VIDEO_COLUMNS].drop_duplicates(subset=["video_id"])
comments = df[COMMENT_COLUMNS]

# NaN -> NULL
videos = videos.astype(object).where(videos.notna(), None)
comments = comments.astype(object).where(comments.notna(), None)

# ==========================================================
# 4. WRITE DATABASE
# ==========================================================
# The database is rebuilt on every run, so it always matches the
# cleaned CSV and running this step twice never duplicates rows.

con = sqlite3.connect(DATABASE_FILE)
try:
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
        DROP VIEW IF EXISTS cleaned_comments;
        DROP TABLE IF EXISTS comments;
        DROP TABLE IF EXISTS videos;
    """)
    con.executescript(SCHEMA)

    with con:
        con.executemany(
            f"INSERT INTO videos VALUES ({', '.join('?' * len(VIDEO_COLUMNS))})",
            videos.itertuples(index=False, name=None),
        )
        con.executemany(
            f"INSERT INTO comments VALUES ({', '.join('?' * len(COMMENT_COLUMNS))})",
            comments.itertuples(index=False, name=None),
        )

    # ======================================================
    # 5. VERIFY
    # ======================================================

    n_videos = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    n_comments = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    total_likes, total_replies = con.execute(
        "SELECT SUM(likes), SUM(replies) FROM comments"
    ).fetchone()

    print("\n" + "=" * 55)
    print("DATABASE SUMMARY")
    print("=" * 55)
    print(f"videos table   : {n_videos:,} rows")
    print(f"comments table : {n_comments:,} rows (CSV: {len(df):,})")
    print(f"Total likes    : {total_likes:,} (CSV: {int(df['likes'].sum()):,})")
    print(f"Total replies  : {total_replies:,} (CSV: {int(df['replies'].sum()):,})")

    if n_comments != len(df) or total_likes != int(df["likes"].sum()):
        raise SystemExit("Database does not match the cleaned CSV.")

    print("\nTop 5 videos by comments (query on the database):")
    top = pd.read_sql_query(
        """
        SELECT v.video_label, COUNT(*) AS comments, SUM(c.likes) AS likes
        FROM comments c JOIN videos v ON v.video_id = c.video_id
        GROUP BY v.video_id ORDER BY comments DESC LIMIT 5
        """,
        con,
    )
    print(top.to_string(index=False))
finally:
    con.close()

print(f"\nSaved: {DATABASE_FILE}")
