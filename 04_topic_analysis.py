import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from common import (
    CLEANED_FILE, TOPIC_K_EVALUATION, TOPIC_KEYWORDS, TOPIC_RESULTS,
    TOPIC_SUMMARY, ensure_dirs, read_csv, require, save_csv, setup_console,
)

setup_console()
ensure_dirs()
require(CLEANED_FILE, "01_cleaning.py")

print("=" * 55)
print("STEP 4 - TOPIC ANALYSIS (TF-IDF + NMF)")
print("=" * 55)

# Topic modeling is exploratory: it groups comments that share words.
# The topics are not ground truth.

N_TOPICS = 5          # starting point; see the k evaluation below
N_KEYWORDS = 10
RANDOM_STATE = 42

df = read_csv(CLEANED_FILE)

# ==========================================================
# 1. STOPWORDS
# ==========================================================
# Comments are multilingual, so English stopwords alone are not enough.
# Function words of the other languages found in the data are added.

MULTILINGUAL_STOPWORDS = {
    # Tetum
    "nia", "iha", "hau", "mak", "ida", "ita", "nee", "neе", "sira", "nian",
    "lai", "mos", "bele", "ema", "ho", "ba", "maka", "sei", "tan", "ne",
    "nebe", "hotu", "deit", "atu", "ona", "imi", "ami", "sai", "fali",
    "hanesan", "dehan", "dbn", "doben", "laos", "hein", "nafatin", "liu",
    "mai", "tebes", "barak", "krk", "hodi", "ka", "loos", "mt", "kona",
    # Indonesian / Malay
    "dari", "dan", "yang", "ada", "ini", "itu", "saya", "aku", "kami",
    "kita", "juga", "untuk", "kalau", "kalo", "bisa", "apa", "aja", "tapi",
    "ngga", "gak", "nggak", "buat", "lebih", "bukan", "mereka", "anda",
    "sama", "udah", "sudah", "jadi", "dengan", "ke", "di", "ya", "kak",
    "kaka", "bang", "bro", "tau", "orang", "terlalu", "ijin", "izin",
    # Portuguese / Spanish
    "que", "por", "los", "las", "para", "del", "con", "como", "todo",
    "todos", "pero", "más", "mas", "com", "não", "nos", "eso", "una", "uma",
    "ser", "esta", "está", "sin", "tem", "foi", "mais", "nem", "quem", "fue",
    "ver", "tanto", "solo", "nós", "eles", "são", "porque", "gente", "tiene",
    "est", "agora", "muito", "muy", "hay", "ese", "esa", "este", "ele",
    "ela", "mi", "tu", "su", "lo", "le", "el", "la", "de", "en", "es", "se",
    "da", "do", "dos", "das", "um", "al", "ya", "si", "sim", "yo", "vos",
    # Laughter / fillers
    "kkkk", "jajaja", "haha", "hahaha", "wkwk", "wkwkwk",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | MULTILINGUAL_STOPWORDS)

# ==========================================================
# 2. TF-IDF
# ==========================================================

text_df = df[df["has_text"]].copy()

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=STOPWORDS,
    token_pattern=r"(?u)\b[^\W\d_]{3,}\b",   # words of 3+ letters
    min_df=3,          # word must appear in at least 3 comments
    max_df=0.5,        # ignore words in more than 50% of comments
    max_features=2000,
)
X = vectorizer.fit_transform(text_df["clean_comment"])
features = vectorizer.get_feature_names_out()

# Emoji-only or very short comments have no remaining words.
has_terms = np.asarray(X.sum(axis=1)).ravel() > 0
X_model = X[has_terms]

print(f"\nComments with text        : {len(text_df):,}")
print(f"Comments with topic words : {has_terms.sum():,}")
print(f"Vocabulary size           : {len(features):,}")

# ==========================================================
# 3. EVALUATE NUMBER OF TOPICS
# ==========================================================
# Reconstruction error always falls when k grows, so it is only a guide.
# Keyword overlap between topics shows when topics start repeating.


def fit_nmf(k):
    model = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE,
                max_iter=500)
    W = model.fit_transform(X_model)
    return model, W


def top_words(model, n=N_KEYWORDS):
    return [[features[i] for i in comp.argsort()[::-1][:n]]
            for comp in model.components_]


rows = []
for k in range(3, 9):
    model, _ = fit_nmf(k)
    words = top_words(model)
    sets = [set(w) for w in words]
    overlap = np.mean([len(a & b) / N_KEYWORDS
                       for i, a in enumerate(sets) for b in sets[i + 1:]])
    rows.append({"k": k,
                 "reconstruction_error": round(model.reconstruction_err_, 3),
                 "avg_keyword_overlap": round(overlap, 3)})
k_eval = pd.DataFrame(rows)

print("\nNumber-of-topics evaluation:")
print(k_eval.to_string(index=False))

# ==========================================================
# 4. FINAL MODEL
# ==========================================================

nmf, W = fit_nmf(N_TOPICS)
keywords = top_words(nmf)

# Human-readable names are only given when the keywords clearly support
# them. The rule checks the actual keywords; otherwise "Topic N" is kept.
NAME_RULES = [
    ({"ronaldo", "cristiano", "portugal", "messi", "pessi", "copa", "mundial",
      "goat", "fifa"}, "Football / Portugal & Ronaldo"),
    ({"amen", "amin", "rogai", "tuhan", "deus", "god", "jesus", "maria"},
     "Religion / Prayer"),                     # rogai = pray (Tetum)
    ({"timor", "leste", "timur", "east", "katolik", "puro"},
     "Timor-Leste identity / pride"),
    ({"editor", "edit", "stiker", "share", "post"},
     "Video editing / content"),
    ({"furak", "demais", "dmais", "fatin", "bonita", "lindo", "cantik",
      "indah", "suai"},
     "Praise of places / beauty"),             # furak = beautiful, fatin = place
    ({"indonesia", "papua", "merdeka", "negara", "otsus", "ntt"},
     "Indonesia / Papua politics"),
]


def topic_name(topic_number, words):
    top = set(words[:5])
    best, best_hits = None, 0
    for vocab, name in NAME_RULES:
        hits = len(top & vocab)
        if hits > best_hits:
            best, best_hits = name, hits
    # At least 2 of the top-5 keywords must match the rule.
    if best_hits >= 2:
        return f"Topic {topic_number}: {best}"
    return f"Topic {topic_number}"


keyword_rows = []
print("\n" + "=" * 55)
print(f"DISCOVERED TOPICS (k = {N_TOPICS})")
print("=" * 55)
for i, words in enumerate(keywords, start=1):
    name = topic_name(i, words)
    keyword_rows.append({"topic_id": i, "topic_name": name,
                         "keywords": ", ".join(words)})
    print(f"\n{name}\n  {', '.join(words)}")

keywords_df = pd.DataFrame(keyword_rows)
names = dict(zip(keywords_df["topic_id"], keywords_df["topic_name"]))

# ==========================================================
# 5. ASSIGN TOPICS
# ==========================================================
# topic_id 0 = no topic words left (emoji-only / very short / no text).

text_df["topic_id"] = 0
text_df.loc[text_df.index[has_terms], "topic_id"] = W.argmax(axis=1) + 1
text_df["topic_weight"] = 0.0
text_df.loc[text_df.index[has_terms], "topic_weight"] = W.max(axis=1).round(4)

df["topic_id"] = 0
df["topic_weight"] = 0.0
df.loc[text_df.index, ["topic_id", "topic_weight"]] = (
    text_df[["topic_id", "topic_weight"]]
)
df["topic_id"] = df["topic_id"].astype(int)
df["topic_name"] = df["topic_id"].map(names).fillna("No topic (emoji/short/no text)")

summary = (
    df["topic_id"].value_counts().sort_index()
    .rename_axis("topic_id").reset_index(name="comments")
)
summary["topic_name"] = summary["topic_id"].map(names).fillna(
    "No topic (emoji/short/no text)")
assigned = summary.loc[summary["topic_id"] > 0, "comments"].sum()
summary["percentage_of_all"] = (summary["comments"] / len(df) * 100).round(2)
summary["percentage_of_assigned"] = np.where(
    summary["topic_id"] > 0,
    (summary["comments"] / assigned * 100).round(2), np.nan)
summary = summary.merge(keywords_df[["topic_id", "keywords"]],
                        on="topic_id", how="left")

print("\n" + "=" * 55)
print("TOPIC DISTRIBUTION")
print("=" * 55)
print(summary[["topic_id", "topic_name", "comments", "percentage_of_all",
               "percentage_of_assigned"]].to_string(index=False))

print("\nExample comments per topic:")
for t in range(1, N_TOPICS + 1):
    ex = df[df["topic_id"] == t].nlargest(3, "topic_weight")
    print(f"\n{names[t]}")
    for c in ex["clean_comment"]:
        print(f"  - {c[:90]}")

# ==========================================================
# 6. SAVE
# ==========================================================

save_csv(df[["comment_id", "video_id", "video_label", "comment",
             "clean_comment", "topic_id", "topic_name", "topic_weight"]],
         TOPIC_RESULTS)
save_csv(keywords_df, TOPIC_KEYWORDS)
save_csv(summary, TOPIC_SUMMARY)
save_csv(k_eval, TOPIC_K_EVALUATION)

print(f"\nSaved: {TOPIC_RESULTS}")
print(f"Saved: {TOPIC_KEYWORDS}")
print(f"Saved: {TOPIC_SUMMARY}")
print(f"Saved: {TOPIC_K_EVALUATION}")
