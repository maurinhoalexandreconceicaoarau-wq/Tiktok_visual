import re

import pandas as pd

from common import (
    ENGAGEMENT_BY_LENGTH, ENGAGEMENT_COMMENTS, ENGAGEMENT_CORRELATION,
    ENGAGEMENT_VIDEOS, REPORTS_DIR, ROOT, SENTIMENT_BY_LANGUAGE,
    SENTIMENT_BY_VIDEO, SENTIMENT_SUMMARY, TOPIC_SUMMARY, ensure_dirs,
    read_csv, require, save_csv, setup_console, shorten,
)

setup_console()
ensure_dirs()
for path, step in [(ENGAGEMENT_COMMENTS, "05_engagement.py"),
                   (SENTIMENT_SUMMARY, "03_sentiment.py"),
                   (TOPIC_SUMMARY, "04_topic_analysis.py")]:
    require(path, step)

# All numbers below are read from the pipeline outputs; nothing is
# hard-coded. Wording is descriptive ("shows", "is associated with"),
# because the analysis is descriptive/correlational.

df = read_csv(ENGAGEMENT_COMMENTS)
videos = read_csv(ENGAGEMENT_VIDEOS)
sent = pd.read_csv(SENTIMENT_SUMMARY)
sent_video = pd.read_csv(SENTIMENT_BY_VIDEO)
sent_lang = pd.read_csv(SENTIMENT_BY_LANGUAGE)
topics = pd.read_csv(TOPIC_SUMMARY)
corr = pd.read_csv(ENGAGEMENT_CORRELATION)
by_length = pd.read_csv(ENGAGEMENT_BY_LENGTH)

lines = []


def out(text=""):
    lines.append(text)
    print(text)


def heading(title):
    out()
    out(title)
    out("-" * len(title))


def get_corr(subset, method):
    row = corr[(corr["subset"] == subset) & (corr["method"] == method)].iloc[0]
    return row["correlation"], row["interpretation"]


n_comments = len(df)
n_videos = df["video_id"].nunique()
total_likes = int(df["likes"].sum())
total_replies = int(df["replies"].sum())
avg_comments = videos["comments"].mean()
median_comments = videos["comments"].median()

out("TIKTOK COMMENT ANALYTICS - INSIGHTS")
out("Audience Engagement, Sentiment, Topic and Behavioral Insights")
out("=" * 62)
out(f"Dataset: {n_comments:,} comments on {n_videos} videos "
    f"({df['date'].min()} to {df['date'].max()})")
out(f"Total comment likes: {total_likes:,} | Total replies: {total_replies:,}")
out("All figures describe this dataset only (observed in this dataset).")

# ----------------------------------------------------------
heading("Insight 1 - Comment volume")
top = videos.loc[videos["comments"].idxmax()]
second = videos.nlargest(2, "comments").iloc[1]
out(f"{top['video_label']} has the most comments: {top['comments']:,} "
    f"({top['comments'] / n_comments * 100:.1f}% of all comments).")
out(f"That is {top['comments'] / avg_comments:.1f}x the average of "
    f"{avg_comments:.1f} comments per video (median {median_comments:.0f}).")
out(f"The second video, {second['video_label']}, has {second['comments']:,} "
    "comments, so comment volume is concentrated in a few videos.")

# ----------------------------------------------------------
heading("Insight 2 - Engagement")
top_e = videos.loc[videos["engagement"].idxmax()]
out(f"{top_e['video_label']} has the highest comment engagement score "
    f"(likes + replies): {top_e['engagement']:,}, which is "
    f"{top_e['engagement'] / videos['engagement'].sum() * 100:.1f}% of the total.")
out(f"It holds {top_e['share_of_total_likes_pct']:.1f}% of all comment likes and "
    f"averages {top_e['engagement_per_comment']:.1f} engagement per comment, "
    f"versus a median of {videos['engagement_per_comment'].median():.2f} "
    "across videos.")
rest = videos[videos["video_id"] != top_e["video_id"]]
out(f"Without this video, the other {len(rest)} videos together have "
    f"{rest['engagement'].sum():,} engagement. Totals for the dataset are "
    "therefore strongly influenced by one video.")

# ----------------------------------------------------------
heading("Insight 3 - Comment popularity")
tc = df.loc[df["likes"].idxmax()]
out(f"The most-liked comment has {tc['likes']:,} likes and "
    f"{tc['replies']:,} replies ({tc['video_label'][:3]}):")
out(f'  "{shorten(tc["comment"], 120)}"')
top10 = df.nlargest(10, "likes")
out(f"The 10 most-liked comments hold {top10['likes'].sum() / total_likes * 100:.1f}% "
    f"of all comment likes; {int((df['likes'] == 0).sum()):,} comments "
    f"({(df['likes'] == 0).mean() * 100:.1f}%) have zero likes.")
out(f"Median likes per comment = {df['likes'].median():.0f}; "
    f"mean = {df['likes'].mean():.2f}. The distribution is highly skewed.")

# ----------------------------------------------------------
heading("Insight 4 - Discussion (likes vs replies)")
p_all, p_all_i = get_corr("all comments", "pearson")
s_all, s_all_i = get_corr("all comments", "spearman")
p_top, _ = get_corr("top-level comments", "pearson")
s_top, s_top_i = get_corr("top-level comments", "spearman")
out(f"Pearson r (all comments) = {p_all:.3f} -> {p_all_i} linear association.")
out(f"Spearman rho (all comments) = {s_all:.3f} -> {s_all_i}; "
    f"top-level comments only: Pearson {p_top:.3f}, Spearman {s_top:.3f}.")
out("Comments with more likes tend to also receive more replies, but the rank "
    "correlation is weaker than Pearson, which indicates the linear value is "
    "partly driven by a few very popular comments. Correlation does not show "
    "that likes cause replies (or the reverse).")

# ----------------------------------------------------------
heading("Insight 5 - Sentiment")
s = sent.set_index("sentiment")
dom = s["comments"].idxmax()
classified = int(s["comments"].sum())
out(f"Sentiment classification of {classified:,} comments with text "
    f"(VADER + supplementary multilingual lexicon):")
for label in ["Positive", "Neutral", "Negative"]:
    out(f"  {label:<8} {int(s.loc[label, 'comments']):>5,}  "
        f"({s.loc[label, 'percentage']:.1f}%)")
out(f"{dom} is the dominant label.")
sv = sent_video[sent_video["total"] >= 30]
lo = sv.loc[sv["negative_pct"].idxmax()]
hi = sv.loc[sv["positive_pct"].idxmax()]
out(f"Among videos with at least 30 text comments, {hi['video']} has the highest "
    f"positive share ({hi['positive_pct']:.1f}%), and {lo['video']} the highest "
    f"negative share ({lo['negative_pct']:.1f}%).")
no_signal = (~df.loc[df["has_text"], "sentiment_signal"].astype(bool)).mean() * 100
out(f"Limitation: {no_signal:.1f}% of text comments contain no word or emoji the "
    "classifier recognises and are labelled Neutral; the share is highest for "
    f"'Other' languages ({sent_lang.set_index('language_group').loc['Other', 'no_signal_pct']:.1f}%). "
    "The classifier assigns labels based on the selected lexicon method; it "
    "does not understand every comment.")

# ----------------------------------------------------------
heading("Insight 6 - Topics")
assigned = topics[topics["topic_id"] > 0]
dom_t = assigned.loc[assigned["comments"].idxmax()]
none = topics[topics["topic_id"] == 0]
out(f"{int(assigned['comments'].sum()):,} comments contain topic words; "
    f"{int(none['comments'].sum()) if len(none) else 0:,} are emoji-only, very "
    "short or without text and have no topic.")
for _, t in assigned.sort_values("comments", ascending=False).iterrows():
    out(f"  {t['topic_name']:<42} {int(t['comments']):>4,} "
        f"({t['percentage_of_assigned']:.1f}%)  [{t['keywords']}]")
out(f"The most frequent topic is {dom_t['topic_name']} "
    f"({dom_t['percentage_of_assigned']:.1f}% of comments with a topic).")
topic_eng = (df[df["topic_id"] > 0].groupby("topic_name")["engagement"]
             .agg(["mean", "median"]).sort_values("mean", ascending=False))
best = topic_eng.index[0]
out(f"{best} has the highest average engagement per comment "
    f"({topic_eng.iloc[0]['mean']:.1f}; median {topic_eng.iloc[0]['median']:.0f}).")
out("Topics come from unsupervised TF-IDF + NMF and are exploratory, "
    "not ground-truth categories.")

# ----------------------------------------------------------
heading("Insight 7 - Audience behavior")
text_df = df[df["has_text"]]
letters = text_df["clean_comment"].str.contains(r"[^\W\d_]", regex=True)
out(f"Comments are short: median {text_df['comment_length'].median():.0f} "
    f"characters, average {text_df['comment_length'].mean():.1f}; "
    f"{(~letters).mean() * 100:.1f}% of text comments are emoji/symbol-only.")
short = by_length.iloc[0]
best_len = by_length.loc[by_length["avg_engagement"].idxmax()]
len_r, _ = get_corr("comments with text", "spearman (length vs engagement)")
out(f"Comments of {short['length_group']} characters average "
    f"{short['avg_engagement']:.2f} engagement ({short['pct_with_any_like']:.1f}% "
    f"receive any like), while {best_len['length_group']}-character comments "
    f"average {best_len['avg_engagement']:.2f} "
    f"({best_len['pct_with_any_like']:.1f}% receive any like).")
out(f"Spearman correlation between length and engagement = {len_r:.3f}: longer "
    "comments are associated with somewhat higher engagement in this dataset, "
    "up to a point (the longest group averages "
    f"{by_length.iloc[-1]['avg_engagement']:.2f}). Averages are pulled up by a "
    "few viral comments, so the share of comments with any like is the more "
    "robust comparison.")
reply_share = df["is_reply"].mean() * 100
repeat = int(df["is_repeated_comment"].sum())
out(f"{reply_share:.1f}% of rows are replies to other comments, and {repeat:,} "
    "comments repeat the same text by the same user on the same video.")

out()
out("Note: engagement here is the comment engagement score (comment likes + "
    "replies). Video views, shares and video likes are not in the dataset.")

# ==========================================================
# SAVE
# ==========================================================

(REPORTS_DIR / "insights.txt").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")

top_by = {m: videos.loc[videos[m].idxmax(), "video_label"]
          for m in ["comments", "total_likes", "total_replies", "engagement"]}
summary = pd.DataFrame([
    ("total_comments", n_comments),
    ("total_videos", n_videos),
    ("total_comment_likes", total_likes),
    ("total_replies", total_replies),
    ("avg_likes_per_comment", round(df["likes"].mean(), 3)),
    ("avg_replies_per_comment", round(df["replies"].mean(), 3)),
    ("comments_with_text", classified),
    ("positive_pct", s.loc["Positive", "percentage"]),
    ("neutral_pct", s.loc["Neutral", "percentage"]),
    ("negative_pct", s.loc["Negative", "percentage"]),
    ("dominant_topic", dom_t["topic_name"]),
    ("top_video_by_comments", top_by["comments"]),
    ("top_video_by_likes", top_by["total_likes"]),
    ("top_video_by_replies", top_by["total_replies"]),
    ("top_video_by_engagement", top_by["engagement"]),
    ("pearson_likes_replies", p_all),
    ("spearman_likes_replies", s_all),
    ("avg_comment_length_chars", round(text_df["comment_length"].mean(), 2)),
    ("median_comment_length_chars", text_df["comment_length"].median()),
    ("first_comment_date", df["date"].min()),
    ("last_comment_date", df["date"].max()),
], columns=["metric", "value"])
save_csv(summary, REPORTS_DIR / "summary.csv")

# ==========================================================
# UPDATE README DATASET SECTION (computed, not typed by hand)
# ==========================================================

readme = ROOT / "README.md"
if readme.exists():
    fields = ", ".join(f"`{c}`" for c in pd.read_csv(
        ROOT / "data" / "tiktok_comments.csv", nrows=0).columns)
    block = (
        "<!-- DATASET_STATS_START -->\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Comments | {n_comments:,} |\n"
        f"| Videos | {n_videos} |\n"
        f"| Unique commenters | {df['username'].nunique():,} |\n"
        f"| Total comment likes | {total_likes:,} |\n"
        f"| Total replies | {total_replies:,} |\n"
        f"| Comment dates | {df['date'].min()} to {df['date'].max()} |\n"
        f"| Positive / Neutral / Negative | "
        f"{s.loc['Positive', 'percentage']:.1f}% / "
        f"{s.loc['Neutral', 'percentage']:.1f}% / "
        f"{s.loc['Negative', 'percentage']:.1f}% |\n"
        f"| Pearson r (likes vs replies) | {p_all:.3f} |\n\n"
        f"Raw fields: {fields}\n\n"
        "_This table is generated by `07_insights.py`._\n"
        "<!-- DATASET_STATS_END -->"
    )
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"<!-- DATASET_STATS_START -->.*?<!-- DATASET_STATS_END -->",
                  lambda _: block, text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
    print("\nREADME dataset section updated.")

print(f"\nSaved: {REPORTS_DIR / 'insights.txt'}")
print(f"Saved: {REPORTS_DIR / 'summary.csv'}")
