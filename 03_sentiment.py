import re

import pandas as pd

from common import (
    CLEANED_FILE, NO_TEXT_LABEL, SENTIMENT_BY_LANGUAGE, SENTIMENT_BY_VIDEO,
    SENTIMENT_ORDER, SENTIMENT_RESULTS, SENTIMENT_SUMMARY, ensure_dirs,
    read_csv, require, save_csv, setup_console,
)

setup_console()
ensure_dirs()
require(CLEANED_FILE, "01_cleaning.py")

print("=" * 55)
print("STEP 3 - SENTIMENT CLASSIFICATION")
print("=" * 55)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    raise SystemExit(
        "vaderSentiment is not installed. Run: pip install -r requirements.txt"
    )

df = read_csv(CLEANED_FILE)

# ==========================================================
# 1. LANGUAGE / TEXT INSPECTION
# ==========================================================
# The 'language' column is TikTok's own language tag ('un' = undetermined).

text_df = df[df["has_text"]]
letters = text_df["clean_comment"].str.contains(r"[^\W\d_]", regex=True)

print(f"\nComments with text: {len(text_df):,} of {len(df):,}")
print(f"Emoji/symbol-only comments: {(~letters).sum():,} "
      f"({(~letters).mean() * 100:.1f}% of comments with text)")

lang = text_df["language"].value_counts(normalize=True).mul(100).round(1)
print("\nTop languages (% of comments with text):")
print(lang.head(8).to_string())
print(f"\nEnglish share: {lang.get('en', 0):.1f}%  ->  the comments are "
      "multilingual (Tetum, Indonesian, Spanish, Portuguese, English, emoji).")

# ==========================================================
# 2. METHOD
# ==========================================================
# VADER is a lexicon-based classifier built for English social media
# text. It also scores emoji, which make up a large part of this dataset.
# Because most comments are not English, VADER's lexicon is extended with
# a SMALL, documented list of sentiment words that actually appear in
# this dataset (Tetum, Indonesian, Portuguese, Spanish). Scores use
# VADER's own scale (-4 very negative ... +4 very positive).
#
# Limitation: words outside the lexicon are ignored, so many non-English
# comments are classified as Neutral. The classifier assigns labels based
# on these word lists; it does not understand every comment.

SUPPLEMENTARY_LEXICON = {
    # Tetum
    "furak": 2.5,        # beautiful
    "diak": 1.9,         # good
    "hadomi": 2.3,       # love
    "susar": -1.2,       # difficult
    # Indonesian / Malay
    "keren": 2.0,        # cool
    "terbaik": 2.5,      # the best
    "mantap": 2.0,       # great
    "bagus": 2.0,        # good
    "cantik": 2.2,       # beautiful
    "indah": 2.2,        # beautiful
    "hebat": 2.2,        # great
    "terimakasih": 1.5,  # thank you
    "makasih": 1.5,      # thanks
    "sakit": -1.5,       # sick / hurt
    "marah": -2.0,       # angry
    # Portuguese / Spanish
    "bonita": 2.5,       # beautiful
    "lindo": 2.5,        # beautiful
    "linda": 2.5,        # beautiful
    "obrigado": 1.5,     # thank you
    "obrigada": 1.5,     # thank you
    "viva": 1.5,         # long live
    "mejor": 1.9,        # best / better
    "orgulho": 2.0,      # pride
    "feliz": 2.2,        # happy
    "amor": 2.3,         # love
    "triste": -2.1,      # sad
    # Slang
    "goat": 1.8,         # "greatest of all time"
}

# English lexicon entries that are false matches in this dataset:
# "tia" = aunt (VADER: "thanks in advance"), "sin" = Spanish "without".
NEUTRALIZED_WORDS = ["tia", "sin"]

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(SUPPLEMENTARY_LEXICON)
for word in NEUTRALIZED_WORDS:
    analyzer.lexicon.pop(word, None)

# On TikTok the fire emoji is used as praise; VADER reads it as "fire"
# (negative). It is mapped to a positive description instead.
analyzer.emojis["🔥"] = "awesome"

# Variation selectors / skin tones stop VADER from recognising emoji
# such as ❤️ or 🙏🏻, so they are removed before scoring.
MODIFIERS = re.compile("[️\U0001F3FB-\U0001F3FF]")


def score(text):
    return analyzer.polarity_scores(MODIFIERS.sub("", text))["compound"]


def label(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


print("\nMethod: VADER compound score + supplementary multilingual lexicon")
print("Thresholds: compound >= 0.05 Positive | <= -0.05 Negative | else Neutral")

# ==========================================================
# 3. CLASSIFY
# ==========================================================

df["sentiment_score"] = 0.0
df.loc[df["has_text"], "sentiment_score"] = (
    df.loc[df["has_text"], "clean_comment"].apply(score)
)
df["sentiment"] = df["sentiment_score"].apply(label)
df.loc[~df["has_text"], "sentiment"] = NO_TEXT_LABEL

# Score exactly 0 = no word/emoji from the lexicon was recognised.
df["sentiment_signal"] = df["sentiment_score"].ne(0)

classified = df[df["has_text"]]

# ==========================================================
# 4. SUMMARY
# ==========================================================

summary = (
    classified["sentiment"].value_counts()
    .reindex(SENTIMENT_ORDER, fill_value=0)
    .rename_axis("sentiment").reset_index(name="comments")
)
summary["percentage"] = (summary["comments"] / len(classified) * 100).round(2)

print("\n" + "=" * 55)
print(f"SENTIMENT SUMMARY ({len(classified):,} comments with text)")
print("=" * 55)
print(summary.to_string(index=False))
print(f"\nExcluded (no text): {(~df['has_text']).sum():,}")
print(f"Comments with no recognised sentiment word/emoji: "
      f"{(~classified['sentiment_signal']).sum():,} "
      f"({(~classified['sentiment_signal']).mean() * 100:.1f}%) -> Neutral")

print("\nExamples:")
for s in SENTIMENT_ORDER:
    ex = classified[(classified["sentiment"] == s)
                    & classified["clean_comment"].str.len().between(8, 70)]
    for _, r in ex.sample(min(3, len(ex)), random_state=1).iterrows():
        print(f"  [{s:<8} {r['sentiment_score']:+.3f}] {r['clean_comment']}")

# ==========================================================
# 5. SENTIMENT BY VIDEO
# ==========================================================

by_video = (
    pd.crosstab(classified["video_label"], classified["sentiment"])
    .reindex(columns=SENTIMENT_ORDER, fill_value=0)
)
by_video.columns = [c.lower() for c in by_video.columns]
by_video["total"] = by_video.sum(axis=1)
for c in ["positive", "neutral", "negative"]:
    by_video[f"{c}_pct"] = (by_video[c] / by_video["total"] * 100).round(2)
by_video = by_video.sort_values("total", ascending=False).reset_index()
by_video = by_video.rename(columns={"video_label": "video"})

print("\nSENTIMENT BY VIDEO (top 10 by comments)")
print(by_video.head(10)[["video", "positive", "neutral", "negative",
                         "positive_pct"]].to_string(index=False))

# ==========================================================
# 6. SENTIMENT BY LANGUAGE GROUP (shows the method's coverage)
# ==========================================================

groups = {"en": "English", "id": "Indonesian/Malay", "ms": "Indonesian/Malay",
          "pt": "Portuguese/Spanish", "es": "Portuguese/Spanish",
          "gl": "Portuguese/Spanish", "un": "Undetermined (mostly emoji)"}
lang_group = classified["language"].map(groups).fillna("Other")

by_language = (
    pd.crosstab(lang_group, classified["sentiment"])
    .reindex(columns=SENTIMENT_ORDER, fill_value=0)
)
by_language["total"] = by_language.sum(axis=1)
by_language["no_signal_pct"] = (
    (~classified["sentiment_signal"]).groupby(lang_group).mean() * 100
).round(2)
by_language = by_language.sort_values("total", ascending=False).reset_index()
by_language = by_language.rename(columns={"language": "language_group"})

print("\nSENTIMENT BY LANGUAGE GROUP")
print(by_language.to_string(index=False))

# ==========================================================
# 7. SAVE
# ==========================================================

result_cols = ["comment_id", "video_id", "video_label", "comment",
               "clean_comment", "language", "has_text", "sentiment",
               "sentiment_score", "sentiment_signal"]
save_csv(df[result_cols], SENTIMENT_RESULTS)
save_csv(summary, SENTIMENT_SUMMARY)
save_csv(by_video, SENTIMENT_BY_VIDEO)
save_csv(by_language, SENTIMENT_BY_LANGUAGE)

print(f"\nSaved: {SENTIMENT_RESULTS}")
print(f"Saved: {SENTIMENT_SUMMARY}")
print(f"Saved: {SENTIMENT_BY_VIDEO}")
print(f"Saved: {SENTIMENT_BY_LANGUAGE}")
