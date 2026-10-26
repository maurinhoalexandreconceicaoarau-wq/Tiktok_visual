"""
Shared paths and helpers used by every step of the pipeline.

Keeping these in one place guarantees that all scripts read and write the
same files and use the same column names.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_JSON = DATA_DIR / "tiktok_comments.json"
RAW_CSV = DATA_DIR / "tiktok_comments.csv"

OUTPUT_DIR = ROOT / "output"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
SENTIMENT_DIR = OUTPUT_DIR / "sentiment"
TOPICS_DIR = OUTPUT_DIR / "topics"
ENGAGEMENT_DIR = OUTPUT_DIR / "engagement"
VISUALIZATION_DIR = OUTPUT_DIR / "visualization"
REPORTS_DIR = OUTPUT_DIR / "reports"
DATABASE_DIR = OUTPUT_DIR / "database"

DATABASE_FILE = DATABASE_DIR / "tiktok_comments.db"
CLEANED_FILE = CLEANED_DIR / "cleaned_comments.csv"
SENTIMENT_RESULTS = SENTIMENT_DIR / "sentiment_results.csv"
SENTIMENT_SUMMARY = SENTIMENT_DIR / "sentiment_summary.csv"
SENTIMENT_BY_VIDEO = SENTIMENT_DIR / "sentiment_by_video.csv"
SENTIMENT_BY_LANGUAGE = SENTIMENT_DIR / "sentiment_by_language.csv"
TOPIC_RESULTS = TOPICS_DIR / "topic_results.csv"
TOPIC_KEYWORDS = TOPICS_DIR / "topic_keywords.csv"
TOPIC_SUMMARY = TOPICS_DIR / "topic_summary.csv"
TOPIC_K_EVALUATION = TOPICS_DIR / "topic_k_evaluation.csv"
ENGAGEMENT_COMMENTS = ENGAGEMENT_DIR / "engagement_comments.csv"
ENGAGEMENT_VIDEOS = ENGAGEMENT_DIR / "engagement_videos.csv"
ENGAGEMENT_CORRELATION = ENGAGEMENT_DIR / "engagement_correlation.csv"
ENGAGEMENT_BY_LENGTH = ENGAGEMENT_DIR / "engagement_by_length.csv"
TOP_COMMENTS = ENGAGEMENT_DIR / "top_comments.csv"

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
NO_TEXT_LABEL = "No Text"

# IDs are long numbers; read them as text so they are never rounded.
ID_COLUMNS = ["comment_id", "video_id"]


def ensure_dirs():
    for d in [CLEANED_DIR, SENTIMENT_DIR, TOPICS_DIR, ENGAGEMENT_DIR,
              VISUALIZATION_DIR, REPORTS_DIR, DATABASE_DIR]:
        os.makedirs(d, exist_ok=True)


def setup_console():
    """Comments contain emoji; avoid UnicodeEncodeError on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def require(path, step):
    """Stop with a clear message when an earlier step has not been run."""
    if not Path(path).exists():
        raise SystemExit(
            f"Missing file: {path}\nRun {step} first."
        )


def read_csv(path, **kwargs):
    dtype = {c: str for c in ID_COLUMNS}
    df = pd.read_csv(path, dtype=dtype, **kwargs)
    # Empty strings are read back as NaN; text columns should stay "".
    for col in ["comment", "clean_comment", "video_caption"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ==========================================================
# COLUMN DETECTION
# ==========================================================

# Candidate names are checked in order; the first one found in the
# dataset is used. The standard name on the left is what every later
# script uses.
COLUMN_CANDIDATES = {
    "comment_id": ["commentId", "comment_id", "cid", "id"],
    "video_id": ["videoId", "video_id", "aweme_id"],
    "video_url": ["videoUrl", "video_url", "videoWebUrl"],
    "video_caption": ["videoCaption", "video_caption", "video", "Video"],
    "comment": ["text", "comment", "commentText", "Comment",
                "CommentText"],
    "likes": ["likeCount", "diggCount", "likes", "like_count", "Likes"],
    "replies": ["replyCount", "replyCommentTotal", "replies",
                "reply_count", "Replies"],
    "created_at": ["createdAt", "createTimeISO", "created_at", "date",
                   "createTime"],
    "language": ["language", "lang"],
    "is_reply": ["isReply", "is_reply"],
    "username": ["commenterUsername", "uniqueId", "username", "user"],
    "pinned_by_creator": ["pinnedByCreator"],
    "liked_by_creator": ["likedByCreator"],
}

REQUIRED_COLUMNS = ["comment", "likes", "replies"]


def detect_columns(columns):
    found = {}
    for standard, candidates in COLUMN_CANDIDATES.items():
        for c in candidates:
            if c in columns:
                found[standard] = c
                break
    missing = [c for c in REQUIRED_COLUMNS if c not in found]
    if missing:
        raise ValueError(
            f"Required columns not found: {missing}. "
            f"Available columns: {list(columns)}"
        )
    return found


# ==========================================================
# TEXT HELPERS
# ==========================================================

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[\w.]+")
# Emoji, pictographs, flags and variation selectors (not drawable by
# matplotlib's default font, so they are removed from chart labels).
EMOJI_PATTERN = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
    "\U0000200D\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF]+"
)


def clean_comment_text(text):
    """
    Text used for NLP: URLs and @mentions removed, whitespace
    normalized. Case, punctuation and emoji are kept because the
    sentiment classifier (VADER) uses them; TF-IDF lowercases by itself.
    """
    text = "" if pd.isna(text) else str(text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def plain_label(text, max_len=45):
    text = "" if pd.isna(text) else str(text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def shorten(text, max_len=100):
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def engagement_score(likes, replies):
    """
    Comment engagement score = comment likes + replies.
    This is a simple comment-level score, not TikTok's video
    engagement rate (views, shares and video likes are not in the data).
    """
    return likes + replies
