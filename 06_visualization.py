import matplotlib

matplotlib.use("Agg")  # save files only, no window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import (
    ENGAGEMENT_COMMENTS, ENGAGEMENT_CORRELATION, ENGAGEMENT_VIDEOS,
    REPORTS_DIR, SENTIMENT_BY_VIDEO, SENTIMENT_ORDER, SENTIMENT_SUMMARY,
    TOPIC_SUMMARY, VISUALIZATION_DIR, ensure_dirs, read_csv, require,
    setup_console,
)

setup_console()
ensure_dirs()
for path, step in [(ENGAGEMENT_COMMENTS, "05_engagement.py"),
                   (ENGAGEMENT_VIDEOS, "05_engagement.py"),
                   (SENTIMENT_SUMMARY, "03_sentiment.py"),
                   (TOPIC_SUMMARY, "04_topic_analysis.py")]:
    require(path, step)

print("=" * 55)
print("STEP 6 - VISUALIZATION")
print("=" * 55)

# ==========================================================
# STYLE
# ==========================================================
# One accent color for single-series charts; sentiment uses a blue/red
# pair with a gray midpoint (Neutral); topics use a fixed category order.

BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

SENTIMENT_COLORS = {"Positive": "#2a78d6", "Neutral": "#b4b2a9",
                    "Negative": "#e34948"}
TOPIC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NO_TOPIC_COLOR = "#c3c2b7"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#f0f5fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.8,
})

saved = []


def save(fig, name):
    path = VISUALIZATION_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)
    print(f"[OK] {name}")


def barh_top(data, value_col, title, xlabel, name, n=10):
    top = data.nlargest(n, value_col).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top["video_label"], top[value_col], color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in top[value_col]],
                 padding=4, color=INK_2, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.12)
    save(fig, name)


df = read_csv(ENGAGEMENT_COMMENTS)
videos = read_csv(ENGAGEMENT_VIDEOS)
sent_summary = pd.read_csv(SENTIMENT_SUMMARY)
sent_video = pd.read_csv(SENTIMENT_BY_VIDEO)
topic_summary = pd.read_csv(TOPIC_SUMMARY)
correlation = pd.read_csv(ENGAGEMENT_CORRELATION)

# ==========================================================
# 1-3. TOP VIDEOS
# ==========================================================

barh_top(videos, "comments", "Top 10 videos by number of comments",
         "Comments", "01_top_videos_comments.png")
barh_top(videos, "total_likes", "Top 10 videos by total comment likes",
         "Comment likes", "02_top_videos_likes.png")
barh_top(videos, "total_replies", "Top 10 videos by total replies",
         "Replies", "03_top_videos_replies.png")

# ==========================================================
# 4. SENTIMENT DISTRIBUTION
# ==========================================================

s = sent_summary.set_index("sentiment").reindex(SENTIMENT_ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(s.index, s["comments"],
              color=[SENTIMENT_COLORS[x] for x in s.index], width=0.55)
ax.bar_label(bars, labels=[f"{c:,}  ({p:.1f}%)" for c, p in
                           zip(s["comments"], s["percentage"])],
             padding=4, color=INK_2)
ax.set_title(f"Sentiment distribution ({int(s['comments'].sum()):,} "
             "comments with text)")
ax.set_ylabel("Comments")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.margins(y=0.12)
save(fig, "04_sentiment_distribution.png")

# ==========================================================
# 5. SENTIMENT BY VIDEO (100% stacked, top 12 videos by comments)
# ==========================================================

sv = sent_video.nlargest(12, "total").iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 7))
left = np.zeros(len(sv))
for sentiment in SENTIMENT_ORDER:
    pct = sv[f"{sentiment.lower()}_pct"].to_numpy()
    ax.barh(sv["video"], pct, left=left, color=SENTIMENT_COLORS[sentiment],
            height=0.6, label=sentiment, edgecolor=SURFACE, linewidth=1.5)
    for y, (l, p) in enumerate(zip(left, pct)):
        if p >= 8:
            ax.text(l + p / 2, y, f"{p:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if sentiment != "Neutral" else INK)
    left += pct
ax.set_xlabel("% of comments with text")
ax.set_xlim(0, 100)
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0), frameon=False,
          fontsize=9)
ax.set_title("Sentiment by video (share of comments, top 12 videos)", pad=28)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
save(fig, "05_sentiment_by_video.png")

# ==========================================================
# 6. LIKES vs REPLIES
# ==========================================================

pearson = correlation.query("subset == 'all comments' and method == 'pearson'")
r = pearson["correlation"].iloc[0]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df["likes"], df["replies"], s=18, alpha=0.45, color=BLUE,
           edgecolors="none")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_title("Comment likes vs replies (symlog scale)")
ax.set_xlabel("Comment likes")
ax.set_ylabel("Replies")
ax.grid(True)
ax.set_axisbelow(True)
ax.text(0.02, 0.96, f"Pearson r = {r:.3f}   (n = {len(df):,})",
        transform=ax.transAxes, va="top", color=INK_2)
save(fig, "06_likes_vs_replies.png")

# ==========================================================
# 7. COMMENT LENGTH
# ==========================================================

lengths = df.loc[df["has_text"], "comment_length"]
cap = int(lengths.quantile(0.99))
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(lengths.clip(upper=cap), bins=40, color=BLUE, edgecolor=SURFACE,
        linewidth=1)
ax.axvline(lengths.median(), color=INK_2, linestyle="--", linewidth=1)
ax.text(lengths.median() + 2, ax.get_ylim()[1] * 0.92,
        f"median = {lengths.median():.0f} chars", color=INK_2)
ax.set_title("Comment length distribution")
ax.set_xlabel(f"Characters (values above the 99th percentile, {cap}, "
              "are grouped in the last bin)")
ax.set_ylabel("Comments")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "07_comment_length.png")

# ==========================================================
# 8. TOPIC DISTRIBUTION
# ==========================================================

ts = topic_summary.sort_values("topic_id")
colors = [NO_TOPIC_COLOR if t == 0 else TOPIC_COLORS[(t - 1) % 5]
          for t in ts["topic_id"]]
labels = ts["topic_name"].str.replace(r"^Topic \d+: ", "", regex=True)
labels = [f"T{t}: {l}" if t else "No topic words" for t, l in
          zip(ts["topic_id"], labels)]
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(labels[::-1], ts["comments"].iloc[::-1], color=colors[::-1],
               height=0.6)
ax.bar_label(bars, labels=[f"{c:,} ({p:.1f}%)" for c, p in
                           zip(ts["comments"].iloc[::-1],
                               ts["percentage_of_all"].iloc[::-1])],
             padding=4, color=INK_2, fontsize=9)
ax.set_title("Topic distribution (TF-IDF + NMF, exploratory)")
ax.set_xlabel("Comments")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.margins(x=0.15)
save(fig, "08_topic_distribution.png")

# ==========================================================
# 9. TOPIC vs SENTIMENT (row %)
# ==========================================================

tt = df[(df["topic_id"] > 0) & df["sentiment"].isin(SENTIMENT_ORDER)]
ct = pd.crosstab(tt["topic_name"], tt["sentiment"]).reindex(
    columns=SENTIMENT_ORDER, fill_value=0)
ct = ct.loc[sorted(ct.index, key=lambda x: int(x.split()[1].rstrip(":")))]
pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(pct.to_numpy(), cmap=BLUE_RAMP, vmin=0, vmax=100,
               aspect="auto")
ax.set_xticks(range(len(SENTIMENT_ORDER)), SENTIMENT_ORDER)
ax.set_yticks(range(len(pct)), pct.index)
for i in range(pct.shape[0]):
    for j in range(pct.shape[1]):
        v = pct.iat[i, j]
        ax.text(j, i, f"{v:.0f}%\n(n={ct.iat[i, j]})", ha="center",
                va="center", fontsize=8.5, color="white" if v > 55 else INK)
ax.set_title("Sentiment within each topic (row %)")
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.colorbar(im, ax=ax, label="% of topic comments", shrink=0.8)
save(fig, "09_topic_sentiment.png")

# ==========================================================
# 10. ENGAGEMENT BY VIDEO
# ==========================================================

barh_top(videos, "engagement",
         "Top 10 videos by comment engagement score (likes + replies)",
         "Comment engagement score", "10_video_engagement.png")

# ==========================================================
# 11. COMMENTS OVER TIME (only if dates are valid)
# ==========================================================

dates = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
if dates.notna().mean() > 0.9:
    monthly = (dates.dt.tz_convert(None).dt.to_period("M")
               .value_counts().sort_index())
    monthly = monthly.reindex(pd.period_range(monthly.index.min(),
                                              monthly.index.max(), freq="M"),
                              fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = monthly.index.to_timestamp()
    ax.plot(x, monthly.values, color=BLUE, linewidth=2, marker="o",
            markersize=4)
    peak = monthly.idxmax()
    ax.annotate(f"{monthly.max():,} comments ({peak})",
                xy=(peak.to_timestamp(), monthly.max()),
                xytext=(10, -4), textcoords="offset points", color=INK_2)
    ax.set_title("Comment activity over time (comments per month)")
    ax.set_ylabel("Comments")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    save(fig, "11_comments_over_time.png")
else:
    print("[SKIP] 11_comments_over_time.png - no valid date/time column")

# ==========================================================
# 12. LANGUAGE DISTRIBUTION (from the EDA table)
# ==========================================================

lang_file = REPORTS_DIR / "eda_language.csv"
if lang_file.exists():
    lang = pd.read_csv(lang_file, keep_default_na=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(lang["language"].replace({"un": "un (undetermined)"}),
                   lang["comments"], color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in lang["percentage"]],
                 padding=4, color=INK_2, fontsize=9)
    ax.set_title("Top 10 comment languages (TikTok language tag)")
    ax.set_xlabel("Comments")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.1)
    save(fig, "12_language_distribution.png")

print(f"\n{len(saved)} charts saved in: {VISUALIZATION_DIR}")
