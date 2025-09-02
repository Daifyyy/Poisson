# train_rf_over25.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss, precision_recall_fscore_support

from fbrapi_dataset import build_match_frame

# Nastavení soutěže/sezóny
LEAGUE_ID = 9            # Premier League
SEASON_ID = "2023-2024"  # příklad – uprav dle potřeby

# 1) Dataset
df = build_match_frame(LEAGUE_ID, SEASON_ID)
df = df.sort_values("date").reset_index(drop=True)

# 2) Featury (rolling metriky; raw necháme jako doplňkové)
feature_cols = [c for c in df.columns if c.startswith(("xg_roll5_","xga_roll5_","shots_roll5_","sot_roll5_","gf_roll5_","ga_roll5_","poss_roll5_"))]
# volitelně přidej raw (mohou lehce kontaminovat, ale stále jsou z téže sezóny před zápasem v čase t):
feature_cols += [c for c in df.columns if c in ("xg_H","xga_H","shots_H","sot_H","poss_H","xg_A","xga_A","shots_A","sot_A","poss_A")]

X = df[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))
y = df["over25"].astype(int)

# 3) Train/test (bez leakage – není to time-series split, ale na první iteraci stačí;
#    pro produkci zvaž TimeSeriesSplit)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 4) Model
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
rf.fit(Xtr, ytr)

# 5) Metriky
proba = rf.predict_proba(Xte)[:, 1]
print("Log-loss:", log_loss(yte, np.c_[1-proba, proba]))
print("Brier:", brier_score_loss(yte, proba))
prec, rec, f1, _ = precision_recall_fscore_support(yte, (proba >= 0.5).astype(int), average="binary")
print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

# 6) Ulož model + popis featur
Path("models").mkdir(exist_ok=True)
joblib.dump(rf, "models/rf_over25_fbr.joblib")
(Path("models") / "rf_over25_fbr.features.txt").write_text("\n".join(feature_cols), encoding="utf-8")

print("✅ Model uložen:", "models/rf_over25_fbr.joblib")
