import pandas as pd

from common import (
    CLEANED_FILE, ENGAGEMENT_BY_LENGTH, ENGAGEMENT_COMMENTS,
    ENGAGEMENT_CORRELATION, ENGAGEMENT_VIDEOS, SENTIMENT_RESULTS,
    TOP_COMMENTS, TOPIC_RESULTS, engagement_score, ensure_dirs, read_csv,
    require, save_csv, setup_console, shorten,
)

setup_console()
ensure_dirs()
require(CLEANED_FILE, "01_cleaning.py")
require(SENTIMENT_RESULTS, "03_sentiment.py")
require(TOPIC_RESULTS, "04_topic_analysis.py")

pd.set_option("display.width", 200)

print("=" * 55)
print("STEP 5 - ENGAGEMENT ANALYSIS")
print("=" * 55)

# ==========================================================
# 1. LOAD AND MERGE
# ==========================================================
# The comment-level file combines cleaning, sentiment and topic results,
# so the visualization, insights and dashboard read one table.

df = read_csv(CLEANED_FILE)
sentiment = read_csv(SENTIMENT_RESULTS)[
    ["comment_id", "sentiment", "sentiment_score", "sentiment_signal"]]
topics = read_csv(TOPIC_RESULTS)[
    ["comment_id", "topic_id", "topic_name", "topic_weight"]]

df = df.merge(sentiment, on="comment_id", how="left", validate="1:1")
df = df.merge(topics, on="comment_id", how="left", validate="1:1")

for col in ["likes", "replies"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# ==========================================================
# 2. COMMENT ENGAGEMENT SCORE
# ==========================================================
# comment engagement score = comment likes + replies
# (not a video engagement rate: views/shares are not in the dataset)

df["engagement"] = engagement_score(df["likes"], df["replies"])

assert (df["engagement"] >= 0).all(), "Engagement must not be negative"

print(f"\nComments: {len(df):,}")
print(f"Total comment engagement score: {df['engagement'].sum():,}")

# ==========================================================
# 3. VIDEO LEVEL
# ==========================================================

videos = (
    df.groupby(["video_label", "video_id", "video_url"])
    .agg(comments=("comment_id", "count"),
         total_likes=("likes", "sum"),
         total_replies=("replies", "sum"),
         engagement=("engagement", "sum"))
    .reset_index()
)
videos["avg_likes_per_comment"] = (videos["total_likes"] / videos["comments"]).round(3)
videos["avg_replies_per_comment"] = (videos["total_replies"] / videos["comments"]).round(3)
videos["engagement_per_comment"] = (videos["engagement"] / videos["comments"]).round(3)
videos["share_of_total_likes_pct"] = (
    videos["total_likes"] / videos["total_likes"].sum() * 100).round(2)
videos = videos.sort_values("engagement", ascending=False).reset_index(drop=True)


def show_top(metric, title):
    print(f"\nTOP 5 VIDEOS BY {title}")
    print(videos.nlargest(5, metric)[["video_label", metric]]
          .to_string(index=False))


show_top("comments", "COMMENTS")
show_top("total_likes", "COMMENT LIKES")
show_top("total_replies", "REPLIES")
show_top("engagement", "ENGAGEMENT")
show_top("engagement_per_comment", "ENGAGEMENT PER COMMENT")

# ==========================================================
# 4. TOP COMMENTS
# ==========================================================

top_comments = df.nlargest(10, "engagement")[
    ["video_label", "comment", "likes", "replies", "engagement",
     "sentiment", "topic_name"]]

print("\nTOP 10 COMMENTS BY ENGAGEMENT")
for i, (_, r) in enumerate(top_comments.iterrows(), 1):
    print(f"{i:>2}. {shorten(r['comment'], 80) or '[no text]'}")
    print(f"    likes={r['likes']:,}  replies={r['replies']:,}  "
          f"engagement={r['engagement']:,}  ({r['video_label'][:3]})")

# ==========================================================
# 5. CORRELATION: LIKES vs REPLIES
# ==========================================================
# Pearson measures a linear relationship. Likes are very skewed (a few
# comments have thousands of likes), so Spearman (rank-based) is also
# reported. Reply rows cannot receive replies on TikTok, so the analysis
# is repeated for top-level comments only.
# Correlation shows association, not causation.


def interpret(r):
    a = abs(r)
    strength = ("very weak" if a < 0.2 else "weak" if a < 0.4 else
                "moderate" if a < 0.6 else "strong" if a < 0.8 else "very strong")
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


corr_rows = []
for subset_name, sub in [("all comments", df),
                         ("top-level comments", df[~df["is_reply"]])]:
    for method in ["pearson", "spearman"]:
        r = sub["likes"].corr(sub["replies"], method=method)
        corr_rows.append({"subset": subset_name, "method": method,
                          "n": len(sub), "correlation": round(r, 4),
                          "interpretation": interpret(r)})
correlation = pd.DataFrame(corr_rows)

print("\nCORRELATION: COMMENT LIKES vs REPLIES")
print(correlation.to_string(index=False))

# ==========================================================
# 6. ENGAGEMENT BY COMMENT LENGTH
# ==========================================================

bins = [0, 10, 30, 80, 200, 10_000]
labels = ["1-10", "11-30", "31-80", "81-200", "200+"]
text_df = df[df["has_text"]].copy()
text_df["length_group"] = pd.cut(text_df["comment_length"], bins=bins,
                                 labels=labels)
by_length = (
    text_df.groupby("length_group", observed=False)
    .agg(comments=("comment_id", "count"),
         avg_likes=("likes", "mean"),
         median_likes=("likes", "median"),
         avg_replies=("replies", "mean"),
         avg_engagement=("engagement", "mean"),
         pct_with_any_like=("likes", lambda s: (s > 0).mean() * 100))
    .round(3).reset_index()
)
len_corr = text_df["comment_length"].corr(text_df["engagement"],
                                          method="spearman")

print("\nENGAGEMENT BY COMMENT LENGTH (characters)")
print(by_length.to_string(index=False))
print(f"Spearman correlation length vs engagement: {len_corr:.4f}")

# ==========================================================
# 7. SAVE
# ==========================================================

correlation = pd.concat([correlation, pd.DataFrame([{
    "subset": "comments with text", "method": "spearman (length vs engagement)",
    "n": len(text_df), "correlation": round(len_corr, 4),
    "interpretation": interpret(len_corr)}])], ignore_index=True)

save_csv(df, ENGAGEMENT_COMMENTS)
save_csv(videos, ENGAGEMENT_VIDEOS)
save_csv(top_comments, TOP_COMMENTS)
save_csv(correlation, ENGAGEMENT_CORRELATION)
save_csv(by_length, ENGAGEMENT_BY_LENGTH)

for p in [ENGAGEMENT_COMMENTS, ENGAGEMENT_VIDEOS, TOP_COMMENTS,
          ENGAGEMENT_CORRELATION, ENGAGEMENT_BY_LENGTH]:
    print(f"Saved: {p}")
