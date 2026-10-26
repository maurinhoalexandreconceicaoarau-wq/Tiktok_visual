import json

import pandas as pd

from common import (
    CLEANED_FILE, RAW_CSV, RAW_JSON, clean_comment_text, detect_columns,
    ensure_dirs, plain_label, save_csv, setup_console,
)

setup_console()
ensure_dirs()

print("=" * 55)
print("STEP 1 - DATA CLEANING")
print("=" * 55)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
# The scraper exported JSON. data/tiktok_comments.csv is the main
# dataset file; it is created once from the JSON if it does not exist.

if not RAW_CSV.exists():
    print(f"\n{RAW_CSV.name} not found -> creating it from {RAW_JSON.name}")
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    raw = pd.DataFrame(records)
    # List/dict fields (mentions, hashtags, ...) are stored as JSON text.
    for col in raw.columns:
        if raw[col].map(lambda v: isinstance(v, (list, dict))).any():
            raw[col] = raw[col].map(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict)) else v
            )
    raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

df = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=True)

print(f"\nDataset: {RAW_CSV}")
print(f"Shape: {df.shape}")

original_rows = len(df)

# ==========================================================
# 2. DETECT AND STANDARDIZE COLUMNS
# ==========================================================

column_map = detect_columns(df.columns)

print("\nDetected columns (standard name <- original name):")
for standard, original in column_map.items():
    print(f"  {standard:<18} <- {original}")

missing_raw = df.isna().sum()
missing_raw = missing_raw[missing_raw > 0]

df = df[list(column_map.values())].rename(
    columns={v: k for k, v in column_map.items()}
)

# ==========================================================
# 3. REMOVE COMPLETELY EMPTY ROWS
# ==========================================================

before = len(df)
df = df.dropna(how="all")
empty_rows_removed = before - len(df)

# ==========================================================
# 4. REMOVE DUPLICATE ROWS
# ==========================================================
# A duplicate is the same comment scraped twice (same comment ID, or an
# identical row when there is no ID). Different users writing the same
# text (e.g. "Amen") are NOT duplicates and are kept.

before = len(df)
if "comment_id" in df.columns:
    df = df.drop_duplicates(subset=["comment_id"])
else:
    df = df.drop_duplicates()
duplicates_removed = before - len(df)

# ==========================================================
# 5. NUMERIC COLUMNS
# ==========================================================
# replies is empty for rows that are themselves replies (TikTok only
# counts replies on top-level comments), so missing -> 0.

for col in ["likes", "replies"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

for col in ["is_reply", "pinned_by_creator", "liked_by_creator"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().eq("true")

# ==========================================================
# 6. TEXT
# ==========================================================

missing_comments = int(df["comment"].isna().sum())

df["comment"] = df["comment"].fillna("").astype(str).str.strip()
df["clean_comment"] = df["comment"].apply(clean_comment_text)

# Comments without text (stickers/GIFs, or only an @mention) are kept
# because their likes and replies are real engagement, but they are
# excluded from sentiment and topic analysis.
df["has_text"] = df["clean_comment"].str.len() > 0

df["comment_length"] = df["comment"].str.len()
df["word_count"] = df["comment"].str.split().str.len().fillna(0).astype(int)

# Same user posting the same text on the same video more than once.
# Flagged, not removed: each one is a separate real comment.
if "username" in df.columns and "video_id" in df.columns:
    df["is_repeated_comment"] = (
        df["has_text"]
        & df.duplicated(subset=["video_id", "username", "comment"], keep="first")
    )
else:
    df["is_repeated_comment"] = False

# ==========================================================
# 7. DATE / TIME
# ==========================================================

if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["date"] = df["created_at"].dt.date
    df["year_month"] = df["created_at"].dt.strftime("%Y-%m")

# ==========================================================
# 8. VIDEO LABEL
# ==========================================================
# Some rows have an empty caption although other rows of the same video
# have it, so the caption is filled per video. Videos are numbered by
# comment count (V01 = most comments) to get short, readable labels.

if "video_id" not in df.columns:
    df["video_id"] = df.get("video_caption", "Unknown video").fillna("Unknown video")

if "video_caption" in df.columns:
    df["video_caption"] = df.groupby("video_id")["video_caption"].transform(
        lambda s: s.fillna(s.dropna().iloc[0] if s.notna().any() else "")
    )
else:
    df["video_caption"] = ""

video_rank = df["video_id"].value_counts().index
video_number = {vid: i + 1 for i, vid in enumerate(video_rank)}

df["video_label"] = df.apply(
    lambda r: f"V{video_number[r['video_id']]:02d} · "
              f"{plain_label(r['video_caption'], 40) or r['video_id']}",
    axis=1,
)

# ==========================================================
# 9. SAVE
# ==========================================================

df = df.sort_values(["video_label", "created_at"]).reset_index(drop=True)
save_csv(df, CLEANED_FILE)

rows_removed = original_rows - len(df)

print("\n" + "=" * 55)
print("CLEANING SUMMARY")
print("=" * 55)
print(f"Original rows              : {original_rows:,}")
print(f"Empty rows removed         : {empty_rows_removed:,}")
print(f"Duplicate rows removed     : {duplicates_removed:,}")
print(f"Rows removed (total)       : {rows_removed:,}")
print(f"Final rows                 : {len(df):,}")
print(f"Missing comment text       : {missing_comments:,}")
print(f"Comments without text      : {(~df['has_text']).sum():,} (kept, excluded from NLP)")
print(f"Repeated comments (flagged): {df['is_repeated_comment'].sum():,} (kept)")

print("\nMissing values in the raw dataset:")
print(missing_raw.to_string() if len(missing_raw) else "  none")

print("\nMissing values after cleaning (important columns):")
important = [c for c in ["comment_id", "video_id", "comment", "likes",
                         "replies", "created_at"] if c in df.columns]
print(df[important].isna().sum().to_string())

print(f"\nSaved: {CLEANED_FILE}")
