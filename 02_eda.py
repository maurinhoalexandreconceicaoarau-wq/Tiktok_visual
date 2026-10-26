import pandas as pd

from common import (
    CLEANED_FILE, RAW_CSV, REPORTS_DIR, engagement_score, ensure_dirs,
    read_csv, require, save_csv, setup_console,
)

setup_console()
ensure_dirs()
require(CLEANED_FILE, "01_cleaning.py")

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

df = read_csv(CLEANED_FILE)
raw = pd.read_csv(RAW_CSV, dtype=str)

print("=" * 55)
print("STEP 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ==========================================================
# RAW DATASET STRUCTURE
# ==========================================================

print(f"\nRaw dataset shape: {raw.shape}")
print("\nRaw columns:")
print(list(raw.columns))

print(f"\nCleaned dataset shape: {df.shape}")
print("\nCleaned dataset head:")
print(df[["video_label", "comment", "likes", "replies", "language"]].head())

print("\nCleaned dataset info:")
df.info()

numeric_cols = df.select_dtypes("number").columns.tolist()
text_cols = [c for c in df.select_dtypes(exclude="number").columns
             if c not in ("has_text", "is_reply", "is_repeated_comment",
                          "pinned_by_creator", "liked_by_creator")]
print(f"\nNumeric columns: {numeric_cols}")
print(f"Text columns   : {text_cols}")

df["engagement"] = engagement_score(df["likes"], df["replies"])

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

overview = pd.DataFrame([
    ("total_comments", len(df)),
    ("top_level_comments", int((~df["is_reply"]).sum())),
    ("reply_comments", int(df["is_reply"].sum())),
    ("comments_with_text", int(df["has_text"].sum())),
    ("unique_videos", df["video_id"].nunique()),
    ("unique_commenters", df["username"].nunique()),
    ("total_likes", int(df["likes"].sum())),
    ("total_replies", int(df["replies"].sum())),
    ("average_likes_per_comment", round(df["likes"].mean(), 3)),
    ("average_replies_per_comment", round(df["replies"].mean(), 3)),
    ("median_likes_per_comment", df["likes"].median()),
    ("first_comment_date", str(df["date"].min())),
    ("last_comment_date", str(df["date"].max())),
], columns=["metric", "value"])

print("\n" + "=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(overview.to_string(index=False))

# ==========================================================
# VIDEO STATISTICS
# ==========================================================

video_stats = (
    df.groupby(["video_label", "video_id"])
    .agg(
        comments=("comment_id", "count"),
        total_likes=("likes", "sum"),
        total_replies=("replies", "sum"),
        engagement=("engagement", "sum"),
    )
    .reset_index()
    .sort_values("comments", ascending=False)
)

print("\n" + "=" * 55)
print("VIDEO STATISTICS")
print("=" * 55)
print(video_stats.drop(columns="video_id").to_string(index=False))

for metric, title in [("comments", "COMMENTS"),
                      ("total_likes", "COMMENT LIKES"),
                      ("total_replies", "REPLIES")]:
    print(f"\nTOP 10 VIDEOS BY {title}")
    print(
        video_stats.nlargest(10, metric)[["video_label", metric]]
        .to_string(index=False)
    )

# ==========================================================
# COMMENT LENGTH (comments with text only)
# ==========================================================

length = df.loc[df["has_text"], "comment_length"]
length_stats = pd.DataFrame([
    ("comments_with_text", len(length)),
    ("average_length", round(length.mean(), 2)),
    ("median_length", length.median()),
    ("min_length", length.min()),
    ("max_length", length.max()),
    ("average_words", round(df.loc[df["has_text"], "word_count"].mean(), 2)),
], columns=["metric", "value"])

print("\n" + "=" * 55)
print("COMMENT LENGTH (characters, comments with text)")
print("=" * 55)
print(length_stats.to_string(index=False))

# ==========================================================
# MISSING VALUES
# ==========================================================

missing = pd.DataFrame({
    "column": raw.columns,
    "missing": raw.isna().sum().values,
})
missing["percentage"] = (missing["missing"] / len(raw) * 100).round(2)
missing = missing.sort_values("missing", ascending=False)

print("\n" + "=" * 55)
print("MISSING VALUES (raw dataset)")
print("=" * 55)
print(missing[missing["missing"] > 0].to_string(index=False))

# ==========================================================
# DUPLICATE ANALYSIS
# ==========================================================

text_df = df[df["has_text"]]
duplicates = pd.DataFrame([
    ("duplicate_rows_raw", int(raw.duplicated().sum())),
    ("duplicate_comment_ids", int(df["comment_id"].duplicated().sum())),
    ("duplicate_texts_all_users", int(text_df["comment"].duplicated().sum())),
    ("repeated_same_user_same_video", int(df["is_repeated_comment"].sum())),
], columns=["check", "count"])

print("\n" + "=" * 55)
print("DUPLICATE ANALYSIS")
print("=" * 55)
print(duplicates.to_string(index=False))
print(
    "\nNote: identical texts from different users (e.g. emoji or 'Amen') "
    "are separate comments and are kept."
)
print("\nMost frequent comment texts:")
print(text_df["comment"].value_counts().head(10).to_string())

# ==========================================================
# LANGUAGE (TikTok 'language' field)
# ==========================================================

language = (
    df["language"].fillna("unknown").value_counts()
    .rename_axis("language").reset_index(name="comments")
)
language["percentage"] = (language["comments"] / len(df) * 100).round(2)

print("\n" + "=" * 55)
print("LANGUAGE DISTRIBUTION (top 15, 'un' = undetermined)")
print("=" * 55)
print(language.head(15).to_string(index=False))

# ==========================================================
# SAVE
# ==========================================================

save_csv(overview, REPORTS_DIR / "eda_overview.csv")
save_csv(video_stats, REPORTS_DIR / "eda_video_stats.csv")
save_csv(length_stats, REPORTS_DIR / "eda_comment_length.csv")
save_csv(missing, REPORTS_DIR / "eda_missing_values.csv")
save_csv(duplicates, REPORTS_DIR / "eda_duplicates.csv")
save_csv(language, REPORTS_DIR / "eda_language.csv")

print(f"\nEDA tables saved in: {REPORTS_DIR}")
