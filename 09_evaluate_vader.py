"""Evaluate the frozen VADER sentiment method against human gold labels.

The development partition is reserved for reviewing the supplementary
lexicon. VADER itself is not trained. The holdout test partition is used
only after the lexicon and thresholds in 03_sentiment.py are frozen.
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dashboard"))
import store

from common import (DATABASE_FILE, SENTIMENT_DIR, SENTIMENT_RESULTS,
                    ensure_dirs, read_csv, setup_console)

setup_console()
ensure_dirs()
LABELS = ["Positive", "Neutral", "Negative"]
EVAL_DIR = SENTIMENT_DIR / "evaluation"

if not store.sentiment_labels_reviewed():
    raise SystemExit("Label seidauk konfirma. Reviza no koreje label sira iha Admin > Anotasaun Sentimentu, depois klik konfirmasaun antes avalia.")

with sqlite3.connect(DATABASE_FILE) as con:
    data = pd.read_sql_query(
        """SELECT comment_id, video_id, clean_comment, gold_label
           FROM sentiment_annotations
           WHERE gold_label IN ('Positive','Neutral','Negative')
             AND trim(clean_comment) <> ''""",
        con, dtype={"comment_id": str, "video_id": str})

if data.empty:
    raise SystemExit("Belum ada gold_label. Isi anotasi manusia di Admin > Anotasaun Sentimentu.")
if set(data["gold_label"]) != set(LABELS):
    missing = sorted(set(LABELS) - set(data["gold_label"]))
    raise SystemExit(f"Anotasi belum mencakup semua kelas: {', '.join(missing)}.")

data["clean_comment"] = data["clean_comment"].fillna("").astype(str)
data["video_id"] = data["video_id"].fillna("").astype(str)
empty_video = data["video_id"].eq("")
data.loc[empty_video, "video_id"] = data.loc[empty_video, "comment_id"]
group_counts = data.groupby("gold_label")["video_id"].nunique()
folds = min(5, int(group_counts.min()))
if folds < 2:
    raise SystemExit("Perlu minimal dua video berbeda untuk setiap label agar train/test tidak berbagi video.")

# Pick a deterministic grouped stratified fold with all labels on each side.
splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
split = None
for development_idx, test_idx in splitter.split(data, data["gold_label"], groups=data["video_id"]):
    if (set(data.iloc[development_idx]["gold_label"]) == set(LABELS)
            and set(data.iloc[test_idx]["gold_label"]) == set(LABELS)):
        split = development_idx, test_idx
        break
if split is None:
    raise SystemExit("Tidak dapat membagi data per video dengan tiga kelas pada kedua bagian. Tambahkan anotasi di video lain.")

development = data.iloc[split[0]].copy()
test = data.iloc[split[1]].copy()
store.save_annotation_split(development["comment_id"], test["comment_id"])

if not SENTIMENT_RESULTS.exists():
    raise SystemExit("Hasil VADER belum ada. Bekukan konfigurasi di 03_sentiment.py lalu jalankan: python 03_sentiment.py")
vader = read_csv(SENTIMENT_RESULTS)
if not {"comment_id", "sentiment"}.issubset(vader.columns):
    raise SystemExit(f"File {SENTIMENT_RESULTS} harus memiliki kolom comment_id dan sentiment.")
vader = vader[["comment_id", "sentiment"]].drop_duplicates("comment_id")
scored = test.merge(vader, on="comment_id", how="left", validate="one_to_one")
if scored["sentiment"].isna().any():
    missing_count = int(scored["sentiment"].isna().sum())
    raise SystemExit(f"Prediksi VADER tidak ditemukan untuk {missing_count} baris test. Jalankan ulang python 03_sentiment.py dari data yang sama.")
if not set(scored["sentiment"]).issubset(set(LABELS)):
    raise SystemExit("Kolom sentiment mengandung label di luar Positive, Neutral, Negative.")

truth = scored["gold_label"]
predicted = scored["sentiment"]
metrics = pd.DataFrame([{
    "method": "VADER + supplementary lexicon",
    "accuracy": accuracy_score(truth, predicted),
    "macro_precision": precision_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
    "macro_recall": recall_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
    "macro_f1": f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
    "test_comments": len(scored),
    "test_videos": scored["video_id"].nunique(),
}])
report = classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)
by_class = pd.DataFrame([{
    "label": label,
    "precision": report[label]["precision"],
    "recall": report[label]["recall"],
    "f1": report[label]["f1-score"],
    "support": int(report[label]["support"]),
} for label in LABELS])
matrix = confusion_matrix(truth, predicted, labels=LABELS)
confusion = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
confusion.index.name = "gold_label"
confusion.columns.name = "predicted_label"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
metrics.to_csv(EVAL_DIR / "vader_metrics.csv", index=False, encoding="utf-8-sig")
by_class.to_csv(EVAL_DIR / "vader_metrics_by_class.csv", index=False, encoding="utf-8-sig")
confusion.to_csv(EVAL_DIR / "vader_confusion_matrix.csv", encoding="utf-8-sig")
scored.to_csv(EVAL_DIR / "vader_test_predictions.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
image = ax.imshow(matrix, cmap="Blues")
ax.set_title("VADER — Confusion Matrix")
ax.set_xticks(range(len(LABELS)), LABELS, rotation=20, ha="right")
ax.set_yticks(range(len(LABELS)), LABELS)
ax.set_xlabel("Prediksi VADER")
ax.set_ylabel("Label acuan manusia")
for row in range(len(LABELS)):
    for column in range(len(LABELS)):
        ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
fig.colorbar(image, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(EVAL_DIR / "vader_confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Data pengembangan: {len(development):,} komentar / {development.video_id.nunique():,} video")
print(f"Data test: {len(scored):,} komentar / {scored.video_id.nunique():,} video")
print("Label baris = label acuan; kolom = prediksi VADER")
print(confusion.to_string())
print("\nMetrik keseluruhan")
print(metrics.to_string(index=False))
print("\nMetrik per kelas")
print(by_class.to_string(index=False))
print(f"\nHasil disimpan di: {EVAL_DIR}")
