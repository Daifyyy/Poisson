# train_models_fbr.py
from __future__ import annotations
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split  # pro první běh; do produkce TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss, precision_recall_fscore_support

from fbrapi_dataset import build_three_seasons

# === Parametry ===
LEAGUE_ID = 9
SEASONS = ["2025-2026", "2024-2025", "2023-2024"]  # poslední 3 sezóny vč. aktuální

# === Dataset ===
df = build_three_seasons(LEAGUE_ID, SEASONS).dropna(subset=["over25","outcome"])
df = df.sort_values("date").reset_index(drop=True)

# === Featury ===
roll_feats = [c for c in df.columns if c.startswith(("xg_roll5_","xga_roll5_","shots_roll5_","sot_roll5_","gf_roll5_","ga_roll5_","poss_roll5_"))]
raw_feats  = [c for c in ("xg_H","xga_H","shots_H","sot_H","poss_H","xg_A","xga_A","shots_A","sot_A","poss_A") if c in df.columns]
FEATURES = roll_feats + raw_feats

X = df[FEATURES].copy().fillna(df[FEATURES].median(numeric_only=True))
w = df["sample_weight"].values

# === 1) Over 2.5 (binární) ===
y_over = df["over25"].astype(int).values
Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y_over, w, test_size=0.25, random_state=42, stratify=y_over)

rf_over = RandomForestClassifier(
    n_estimators=900, max_depth=None, min_samples_leaf=3, n_jobs=-1, random_state=42
)
rf_over.fit(Xtr, ytr, sample_weight=wtr)

proba = rf_over.predict_proba(Xte)[:,1]
print("[Over2.5] Log-loss:", log_loss(yte, np.c_[1-proba, proba], sample_weight=wte))
print("[Over2.5] Brier:", brier_score_loss(yte, proba, sample_weight=wte))
prec, rec, f1, _ = precision_recall_fscore_support(yte, (proba>=0.5).astype(int), average="binary", sample_weight=wte)
print(f"[Over2.5] Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

# === 2) 1X2 (multiclass) ===
# outcome ∈ {"H","D","A"}
y_out = df["outcome"].astype("category")
class_order = ["H","D","A"]
y_out = y_out.cat.set_categories(class_order)
Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y_out, w, test_size=0.25, random_state=42, stratify=y_out)

rf_out = RandomForestClassifier(
    n_estimators=1000, max_depth=None, min_samples_leaf=3, n_jobs=-1, random_state=42
)
rf_out.fit(Xtr, ytr, sample_weight=wtr)

proba3 = rf_out.predict_proba(Xte)  # pořadí odpovídá rf_out.classes_
# Log-loss (multiclass)
print("[1X2] Log-loss:", log_loss(yte, proba3, labels=rf_out.classes_, sample_weight=wte))

# simple report
pred = rf_out.predict(Xte)
prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average=None, labels=class_order, sample_weight=wte)
for lab, P,R,F in zip(class_order, prec, rec, f1):
    print(f"[1X2] {lab}: P={P:.3f} R={R:.3f} F1={F:.3f}")

# === Uložení ===
Path("models").mkdir(exist_ok=True)
joblib.dump(rf_over, "models/rf_over25_fbr.joblib")
joblib.dump(rf_out,  "models/rf_1x2_fbr.joblib")
(Path("models") / "features_fbr.txt").write_text("\n".join(FEATURES), encoding="utf-8")

# volitelné: export datasetu
Path("data").mkdir(exist_ok=True)
df.to_csv("data/fbr_matches_3seasons.csv", index=False)
print("✅ Saved: models/rf_over25_fbr.joblib, models/rf_1x2_fbr.joblib, data/fbr_matches_3seasons.csv")
