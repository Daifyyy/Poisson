from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import joblib
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")
DEFAULT_OVER25_MODEL_PATH = Path(__file__).with_name("random_forest_over25_model.joblib")

# Prefer imblearn.Pipeline, fallback na sklearn.Pipeline
try:  # pragma: no cover - optional dependency
    from imblearn.pipeline import Pipeline as _BasePipeline  # type: ignore
except Exception:  # pragma: no cover
    from sklearn.pipeline import Pipeline as _BasePipeline  # type: ignore


class SampleWeightPipeline(_BasePipeline):
    """Pipeline, která propouští sample_weight do posledního estimatoru."""

    def fit(self, X, y=None, sample_weight=None, **fit_params):  # type: ignore[override]
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


# ---------------------------------------------------------------------
# Načtení dat a příprava featur
# ---------------------------------------------------------------------
def _load_matches(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for csv in sorted(data_dir.glob("*_combined_full_updated.csv")):
        df = pd.read_csv(csv)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No training files found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("Date", inplace=True)
    return df


def _compute_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    results = {"H": 1, "D": 0, "A": -1}
    df = df.copy()
    df["home_result"] = df["FTR"].map(results)
    df["away_result"] = -df["home_result"]

    df["home_recent_form"] = (
        df.groupby("HomeTeam")["home_result"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    df["away_recent_form"] = (
        df.groupby("AwayTeam")["away_result"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    return df.drop(columns=["home_result", "away_result"])


def _compute_expected_goals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_xg"] = df.groupby("HomeTeam")["FTHG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_xg"] = df.groupby("AwayTeam")["FTAG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    df["home_conceded"] = df.groupby("HomeTeam")["FTAG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_conceded"] = df.groupby("AwayTeam")["FTHG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["conceded_diff"] = df["home_conceded"] - df["away_conceded"]
    return df


def _compute_elo_difference(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    df = df.copy()
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    ratings = {team: 1500 for team in teams}
    diffs: list[float] = []

    for _, row in df.iterrows():
        h, a, res = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        diffs.append(ratings[h] - ratings[a])
        res_home = 1 if res == "H" else 0.5 if res == "D" else 0
        exp_home = 1 / (1 + 10 ** ((ratings[a] - ratings[h]) / 400))
        ratings[h] += k * (res_home - exp_home)
        ratings[a] += k * ((1 - res_home) - (1 - exp_home))

    df["elo_diff"] = diffs
    return df


def _clip_features(df: pd.DataFrame) -> pd.DataFrame:
    bounds = {
        "home_recent_form": (-1, 1),
        "away_recent_form": (-1, 1),
        "elo_diff": (-400, 400),
        "xg_diff": (-5, 5),
        "home_conceded": (0, 5),
        "away_conceded": (0, 5),
        "conceded_diff": (-5, 5),
        "shots_diff": (-20, 20),
        "shot_target_diff": (-10, 10),
        "corners_diff": (-10, 10),
        "days_since_last_match": (-30, 30),
        "attack_strength_diff": (-2, 2),
        "defense_strength_diff": (-2, 2),
        "home_advantage": (1.0, 1.0),
        "abs_elo_diff": (0, 400),
        "abs_form_diff": (0, 2),
        "abs_xg_diff": (0, 5),
        "abs_shots_diff": (0, 20),
        "abs_shot_target_diff": (0, 10),
        "abs_corners_diff": (0, 10),
        "total_goals_proxy": (0, 5),
        "tempo_proxy": (0, 40),
        "pois_lambda_home": (0.05, 3.5),
        "pois_lambda_away": (0.05, 3.5),
        "pois_p_home": (0.0, 1.0),
        "pois_p_draw": (0.0, 1.0),
        "pois_p_away": (0.0, 1.0),
        "pois_p_over25": (0.0, 1.0),

    }
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df

def _poisson_probs_from_lambdas(lh: float, la: float, max_goals: int = 12) -> Dict[str, float]:
    """
    Spočítá P(Home), P(Draw), P(Away), P(Over2.5) z nezávislých Poisson(λh), Poisson(λa).
    Suma přes mřížku 0..max_goals (stačí 10–12).
    """
    lh = float(max(lh, 1e-6))
    la = float(max(la, 1e-6))
    from math import exp, factorial

    # precompute pmf
    ph = [exp(-lh) * (lh ** k) / factorial(k) for k in range(max_goals + 1)]
    pa = [exp(-la) * (la ** k) / factorial(k) for k in range(max_goals + 1)]

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over25 = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = ph[h] * pa[a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
            if (h + a) > 2:
                p_over25 += p

    # Zbytek mimo mřížku je typicky zanedbatelný (pro λ ~ 0.5–2.5)
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s
    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
    }


def _estimate_match_lambdas(df_row: pd.Series) -> Tuple[float, float]:
    """
    Odhad λh, λa pouze z *minulých* dat (rolling průměry s .shift()) připravené v _prepare_features.
    Konzervativní směs: 50 % 'for' + 50 % 'against' protistrany + lehká domácí výhoda.
    """
    # bezpecne čtení s fallbacky
    h_for = float(df_row.get("home_avg_goals_last5", 1.1))
    a_for = float(df_row.get("away_avg_goals_last5", 1.1))
    h_conc = float(df_row.get("home_goals_against_10", 1.1))
    a_conc = float(df_row.get("away_goals_against_10", 1.1))

    # mírná domácí výhoda ~ +7 % na λh (lze ladit)
    lambda_home = max(0.05, 0.5 * h_for + 0.5 * a_conc) * 1.07
    lambda_away = max(0.05, 0.5 * a_for + 0.5 * h_conc)

    return lambda_home, lambda_away




def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Iterable[str], Any]:
    df = _compute_recent_form(df)
    df = _compute_expected_goals(df)
    df = _compute_elo_difference(df)

    # rolling střely/na branku/rohy
    df["home_shots_avg"] = df.groupby("HomeTeam")["HS"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_shots_avg"] = df.groupby("AwayTeam")["AS"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["shots_diff"] = df["home_shots_avg"] - df["away_shots_avg"]

    df["home_shot_target_avg"] = df.groupby("HomeTeam")["HST"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_shot_target_avg"] = df.groupby("AwayTeam")["AST"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["shot_target_diff"] = df["home_shot_target_avg"] - df["away_shot_target_avg"]

    df["home_corners_avg"] = df.groupby("HomeTeam")["HC"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_corners_avg"] = df.groupby("AwayTeam")["AC"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["corners_diff"] = df["home_corners_avg"] - df["away_corners_avg"]

    # odpočinek (rozdíl dní)
    df["home_last"] = df.groupby("HomeTeam")["Date"].shift()
    df["away_last"] = df.groupby("AwayTeam")["Date"].shift()
    df["home_rest"] = (df["Date"] - df["home_last"]).dt.days
    df["away_rest"] = (df["Date"] - df["away_last"]).dt.days
    df["days_since_last_match"] = df["home_rest"] - df["away_rest"]
    df.drop(columns=["home_last", "away_last", "home_rest", "away_rest"], inplace=True)

    # týmové síly z Poisson utilit
    from utils.poisson_utils.stats import calculate_team_strengths
    attack_strength, defense_strength, _ = calculate_team_strengths(df)
    df["attack_strength_diff"] = df["HomeTeam"].map(attack_strength) - df["AwayTeam"].map(attack_strength)
    df["defense_strength_diff"] = df["HomeTeam"].map(defense_strength) - df["AwayTeam"].map(defense_strength)

    # ... uvnitř _prepare_features po výpočtu attack_strength_diff/defense_strength_diff:

    # Symetrie – menší rozdíly => vyšší prior na remízu
    df["abs_elo_diff"] = df["elo_diff"].abs()
    df["abs_form_diff"] = (df["home_recent_form"] - df["away_recent_form"]).abs()
    df["abs_xg_diff"] = df["xg_diff"].abs()

    # Pokud calculate_team_strengths neumí explicitně „shift před zápas“,
    # použij bezpečnou rolling proxy (bez leakage):
    df["home_goals_for_10"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift().rolling(10, 1).mean())
    df["home_goals_against_10"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.shift().rolling(10, 1).mean())
    df["away_goals_for_10"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift().rolling(10, 1).mean())
    df["away_goals_against_10"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.shift().rolling(10, 1).mean())

    df["gf_diff_10"] = df["home_goals_for_10"] - df["away_goals_for_10"]
    df["ga_diff_10"] = df["home_goals_against_10"] - df["away_goals_against_10"]


    # konstantní home advantage (1.0) – je součástí tréninku
    df["home_advantage"] = 1.0
    
        # --- Symetrické featury (pro lepší zachytávání remíz) ---
    df["abs_elo_diff"] = df["elo_diff"].abs()
    df["abs_form_diff"] = (df["home_recent_form"] - df["away_recent_form"]).abs()
    df["abs_xg_diff"] = df["xg_diff"].abs()
    df["abs_shots_diff"] = df["shots_diff"].abs()
    df["abs_shot_target_diff"] = df["shot_target_diff"].abs()
    df["abs_corners_diff"] = df["corners_diff"].abs()

    # --- Proxy pro gólovost (O/U 2.5) ---
    # rolling průměr celkových gólů v posledních 5 zápasech týmu
    df["home_avg_goals_last5"] = df.groupby("HomeTeam")["FTHG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_avg_goals_last5"] = df.groupby("AwayTeam")["FTAG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["total_goals_proxy"] = df["home_avg_goals_last5"] + df["away_avg_goals_last5"]

    # rolling průměr střel v zápase (HS+AS) – „tempo zápasu“
    df["home_shots_total5"] = df.groupby("HomeTeam")["HS"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_shots_total5"] = df.groupby("AwayTeam")["AS"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["tempo_proxy"] = df["home_shots_total5"] + df["away_shots_total5"]

        # --- Poisson lambdy a pravděpodobnosti (čistě pre-match; vše je se shift()) ---
    # Pokud nejsou sloupce zavedeny, vytvoř je (bez leakage!)
    if "home_avg_goals_last5" not in df.columns:
        df["home_avg_goals_last5"] = df.groupby("HomeTeam")["FTHG"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
    if "away_avg_goals_last5" not in df.columns:
        df["away_avg_goals_last5"] = df.groupby("AwayTeam")["FTAG"].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean()
        )
    if "home_goals_against_10" not in df.columns:
        df["home_goals_against_10"] = df.groupby("HomeTeam")["FTAG"].transform(
            lambda x: x.shift().rolling(10, min_periods=1).mean()
        )
    if "away_goals_against_10" not in df.columns:
        df["away_goals_against_10"] = df.groupby("AwayTeam")["FTHG"].transform(
            lambda x: x.shift().rolling(10, min_periods=1).mean()
        )

    # odhad λh, λa pro každý řádek
    lambdas = df.apply(_estimate_match_lambdas, axis=1)
    df["pois_lambda_home"] = [lh for lh, _ in lambdas]
    df["pois_lambda_away"] = [la for _, la in lambdas]

    # Poisson pravděpodobnosti
    def _poisson_row_probs(row):
        pr = _poisson_probs_from_lambdas(row["pois_lambda_home"], row["pois_lambda_away"], max_goals=12)
        return pd.Series([pr["p_home"], pr["p_draw"], pr["p_away"], pr["p_over25"]],
                         index=["pois_p_home", "pois_p_draw", "pois_p_away", "pois_p_over25"])

    df[["pois_p_home", "pois_p_draw", "pois_p_away", "pois_p_over25"]] = df.apply(_poisson_row_probs, axis=1)


    features = [
        "home_recent_form","away_recent_form","elo_diff","xg_diff",
        "home_conceded","away_conceded","conceded_diff",
        "shots_diff","shot_target_diff","corners_diff",
        "home_advantage","days_since_last_match",
        "attack_strength_diff","defense_strength_diff",

        # symetrie (pokud je máš – ponech klidně i ty)
        "abs_elo_diff","abs_form_diff","abs_xg_diff",
        "abs_shots_diff","abs_shot_target_diff","abs_corners_diff",
        "total_goals_proxy","tempo_proxy",

        # NOVÉ – Poisson
        "pois_lambda_home","pois_lambda_away",
        "pois_p_home","pois_p_draw","pois_p_away","pois_p_over25",
    ]


    X = _clip_features(df[features])
    y_raw = df["FTR"].astype(str)

    mask = X.notna().all(axis=1) & y_raw.notna()
    X = X[mask]
    y_raw = y_raw[mask]

    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    return X, y, features, label_enc


# ---------------------------------------------------------------------
# Trénink modelů
# ---------------------------------------------------------------------
def train_model(
    data_dir: str | Path = "data",
    n_splits: int = 5,
    recent_years: int | None = None,
    n_iter: int = 20,
    max_samples: int | None = None,
    decay_factor: float | None = None,
    param_distributions: Mapping[str, Iterable[Any]] | None = None,
    balance_classes: bool = True,
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train RandomForest (H/D/A) s časovým CV a kalibrací.

    Vrací: (calibrated_model, feature_names, label_encoder, best_cv_score, best_params, per_class_metrics)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.base import clone
    from sklearn.metrics import brier_score_loss, precision_recall_fscore_support

    df = _load_matches(data_dir)
    if recent_years is not None and "Date" in df.columns:
        cutoff = df["Date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["Date"] >= cutoff]
    if max_samples is not None:
        df = df.tail(max_samples)

    X, y, feature_names, label_enc = _prepare_features(df)

    # váhy vzorků podle stáří
    sample_weights = None
    if decay_factor is not None and "Date" in df.columns:
        max_date = df["Date"].max()
        age = (max_date - df.loc[X.index, "Date"]).dt.days.to_numpy()
        sample_weights = np.exp(-decay_factor * age)

    # class_weight (default on)
    class_weight = None
    if balance_classes:
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {cls: w for cls, w in zip(classes, weights)}

    pipeline = SampleWeightPipeline([
        ("model", RandomForestClassifier(
            class_weight=class_weight, random_state=42,
            n_estimators=600, min_samples_leaf=2, max_features="sqrt"
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if param_distributions is None:
        param_distributions = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
        }

    # Optimalizace na neg_log_loss
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        random_state=42,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    fit_params = {"sample_weight": sample_weights} if sample_weights is not None else {}
    search.fit(X, y, **fit_params)

    # Empirický prior z tréninku (v pořadí tříd v y)
    classes_sorted = np.unique(y)
    prior = np.bincount(y, minlength=len(classes_sorted)).astype(float)
    prior = prior / prior.sum()

    best_pipeline = search.best_estimator_
    score = float(-search.best_score_)
    best_params = search.best_params_

    # Kalibrace (isotonic) přes time-CV
    calibrated_model = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=tscv)
    if sample_weights is not None:
        calibrated_model.fit(X, y, sample_weight=sample_weights)
    else:
        calibrated_model.fit(X, y)

    # Ulož prior a pořadí tříd do modelu (k použití v inference)
    calibrated_model.prior_ = prior
    calibrated_model.classes_order_ = classes_sorted

    # -------- OOF predikce pro metriky (BASE -> CALIB -> predikce na te fold) --------
    y_pred = np.empty_like(y)
    y_proba = np.zeros((len(y), len(classes_sorted)))

    for tr, te in tscv.split(X, y):
        base = clone(best_pipeline)
        if sample_weights is not None:
            base.fit(X.iloc[tr], y[tr], sample_weight=sample_weights[tr])
        else:
            base.fit(X.iloc[tr], y[tr])

        calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
        if sample_weights is not None:
            calib.fit(X.iloc[tr], y[tr], sample_weight=sample_weights[tr])
        else:
            calib.fit(X.iloc[tr], y[tr])

        y_pred[te]  = calib.predict(X.iloc[te])
        y_proba[te] = calib.predict_proba(X.iloc[te])

    # Zarovnání pořadí tříd pro metriky:
    # label_enc.classes_ je např. ['A','D','H']; classes_sorted je np.unique(y) ve stejném prostoru
    # Následující výpočet PR bere y_pred vs. y v implicitním pořadí classes_sorted.
    precisions, recalls, _, _ = precision_recall_fscore_support(
        y, y_pred, labels=classes_sorted
    )

    # Přemapuj indexy do pořadí label_enc.classes_ při reportu metrik
    idx_by_label = {cls: i for i, cls in enumerate(classes_sorted)}
    metrics: Dict[str, Dict[str, float]] = {}
    for lbl in label_enc.classes_:
        idx = idx_by_label[label_enc.transform([lbl])[0]]  # index třídy v classes_sorted
        true = (y == classes_sorted[idx]).astype(int)
        prob = y_proba[:, idx]
        metrics[lbl] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "brier": float(brier_score_loss(true, prob)),
        }

    return calibrated_model, feature_names, label_enc, score, best_params, metrics


def train_over25_model(
    data_dir: str | Path = "data",
    n_splits: int = 5,
    recent_years: int | None = None,
    n_iter: int = 20,
    max_samples: int | None = None,
    param_distributions: Mapping[str, Iterable[Any]] | None = None,
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Binary RF pro Over 2.5 (s kalibrací a vyvážením tříd)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import precision_recall_fscore_support, brier_score_loss
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd

    # --- Data ---
    df = _load_matches(data_dir)
    if recent_years is not None and "Date" in df.columns:
        cutoff = df["Date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["Date"] >= cutoff]
    if max_samples is not None:
        df = df.tail(max_samples)

    df["over25"] = np.where(df["FTHG"] + df["FTAG"] > 2.5, "Over 2.5", "Under 2.5")
    X, _, feature_names, _ = _prepare_features(df)
    y_raw = df.loc[X.index, "over25"]

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)  # 0/1 v pořadí label_enc.classes_

    # --- Balanced RF (bez imblearn – robustní fallback) ---
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {cls: w for cls, w in zip(classes, weights)}

    pipeline = Pipeline([
        ("model", RandomForestClassifier(
            class_weight=class_weight, random_state=42,
            n_estimators=600, min_samples_leaf=2, max_features="sqrt"
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if param_distributions is None:
        param_distributions = {
            "model__n_estimators": [100, 200, 300, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
        }

    # --- Hyperparam search na neg_log_loss ---
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        random_state=42,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    search.fit(X, y)

    best_pipeline = search.best_estimator_
    score = float(-search.best_score_)
    best_params = search.best_params_

    # --- Kalibrace (isotonic) přes time-CV ---
    calibrated_model = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=tscv)
    calibrated_model.fit(X, y)

    # --- Ulož empirický prior (v pořadí label_enc.classes_) a pořadí tříd ---
    prior = np.bincount(y, minlength=len(label_enc.classes_)).astype(float)
    prior = prior / prior.sum()
    calibrated_model.prior_ = prior
    # u binární klasifikace 0..1 odpovídá label_enc.classes_
    calibrated_model.classes_order_ = np.arange(len(label_enc.classes_))

    # --- OOF metriky z KALIBROVANÝCH pravděpodobností ---
    y_pred = np.empty_like(y)
    y_proba = np.zeros((len(y), len(classes)))

    for tr, te in tscv.split(X, y):
        base = clone(best_pipeline)
        base.fit(X.iloc[tr], y[tr])

        calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calib.fit(X.iloc[tr], y[tr])

        y_pred[te]  = calib.predict(X.iloc[te])
        y_proba[te] = calib.predict_proba(X.iloc[te])

    precisions, recalls, _, _ = precision_recall_fscore_support(y, y_pred, labels=classes)

    metrics: Dict[str, Dict[str, float]] = {}
    for lbl in label_enc.classes_:
        idx = label_enc.transform([lbl])[0]  # 0 nebo 1
        true = (y == idx).astype(int)
        prob = y_proba[:, idx]
        metrics[lbl] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "brier": float(brier_score_loss(true, prob)),
        }

    return calibrated_model, feature_names, label_enc, score, best_params, metrics


# ---------------------------------------------------------------------
# Uložení/načtení, inference
# ---------------------------------------------------------------------
def save_model(
    model: Any,
    feature_names: Iterable[str],
    label_encoder: Any,
    path: str | Path = DEFAULT_MODEL_PATH,
    best_params: Mapping[str, Any] | None = None,
) -> None:
    joblib.dump({
        "model": model,
        "feature_names": list(feature_names),
        "label_encoder": label_encoder,
        "best_params": best_params or {},
        "prior": getattr(model, "prior_", None),
        "classes_order": getattr(model, "classes_order_", None),
    }, Path(path))


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    try:
        data = joblib.load(Path(path))
        model = data["model"]
        logger.info("Loaded model type: %s", type(model).__name__)
        return model, data["feature_names"], data["label_encoder"], data.get("best_params", {})
    except Exception:
        logger.warning("Training RandomForest model because %s is missing", path)
        model, feature_names, label_enc, _, params, _ = train_model(
            n_splits=2, n_iter=1, max_samples=200
        )
        save_model(model, feature_names, label_enc, path, params)
        return model, feature_names, label_enc, params


def predict_outcome(
    features: Dict[str, float],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    alpha: float = 0.15,
) -> str:
    """Vrať predikovaný výsledek ('H'/'D'/'A') s tlumením pravděpodobností."""
    probs = predict_proba(features, model_path=model_path, alpha=alpha)
    reverse = {"Home Win": "H", "Draw": "D", "Away Win": "A"}
    return reverse[max(probs, key=probs.get)]


def construct_features_for_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: Mapping[str, float],
) -> Dict[str, float]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True)

    def recent_form(team: str) -> float:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        vals: list[float] = []
        for _, r in tm.iterrows():
            if r["HomeTeam"] == team:
                vals.append(1 if r["FTR"] == "H" else 0 if r["FTR"] == "D" else -1)
            else:
                vals.append(1 if r["FTR"] == "A" else 0 if r["FTR"] == "D" else -1)
        return float(np.mean(vals)) if vals else 0.0

    def recent_goals(team: str) -> float:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        goals: list[float] = []
        for _, r in tm.iterrows():
            goals.append(r.get("FTHG", 0) if r["HomeTeam"] == team else r.get("FTAG", 0))
        return float(np.mean(goals)) if goals else 0.0

    def recent_conceded(team: str) -> float:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        goals: list[float] = []
        for _, r in tm.iterrows():
            goals.append(r.get("FTAG", 0) if r["HomeTeam"] == team else r.get("FTHG", 0))
        return float(np.mean(goals)) if goals else 0.0

    def last_match_date(team: str) -> pd.Timestamp | None:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        return None if tm.empty else tm["Date"].max()

    home_form = recent_form(home_team)
    away_form = recent_form(away_team)
    home_goals = recent_goals(home_team)
    away_goals = recent_goals(away_team)
    home_conceded = recent_conceded(home_team)
    away_conceded = recent_conceded(away_team)

    elo_diff = elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)
    xg_diff = home_goals - away_goals
    conceded_diff = home_conceded - away_conceded
    home_advantage = 1.0

    h_last = last_match_date(home_team)
    a_last = last_match_date(away_team)
    days_since_last_match = float((h_last - a_last).days) if h_last and a_last else 0.0

    from utils.poisson_utils.stats import calculate_team_strengths
    attack_strength, defense_strength, _ = calculate_team_strengths(df)
    attack_strength_diff = attack_strength.get(home_team, 0.0) - attack_strength.get(away_team, 0.0)
    defense_strength_diff = defense_strength.get(home_team, 0.0) - defense_strength.get(away_team, 0.0)

    feats = {
        "home_recent_form": home_form,
        "away_recent_form": away_form,
        "elo_diff": float(elo_diff),
        "xg_diff": xg_diff,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded,
        "conceded_diff": conceded_diff,
        "home_advantage": home_advantage,
        "days_since_last_match": days_since_last_match,
        "attack_strength_diff": attack_strength_diff,
        "defense_strength_diff": defense_strength_diff,
    }
    return _clip_features(pd.DataFrame([feats])).iloc[0].to_dict()


def predict_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any]
        | Tuple[Any, Iterable[str], Any, Mapping[str, Any]]
        | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    alpha: float = 0.08,   # shrink k prioru
    beta: float = 0.55,    # váha Poissonu v blendu
) -> Dict[str, float]:
    """Vrať pravděpodobnosti {'Home Win','Draw','Away Win'} v procentech (0–100)."""
    if model_data is None:
        model_data = load_model(model_path)
    model, feature_names, label_enc = model_data[:3]

    prior = getattr(model, "prior_", None)
    classes_order = getattr(model, "classes_order_", None)

    X = _clip_features(pd.DataFrame([features], columns=feature_names))
    rf_probs = model.predict_proba(X)[0]

    # zarovnání
    if classes_order is not None and hasattr(model, "classes_"):
        aligned = np.zeros(len(classes_order))
        for i, cls in enumerate(model.classes_):
            j = list(classes_order).index(cls)
            aligned[j] = rf_probs[i]
        rf_probs = aligned

    # Shrink k prioru
    if prior is None:
        prior = np.ones_like(rf_probs) / len(rf_probs)
    rf_probs = np.clip(rf_probs, 1e-4, 1 - 1e-4)
    rf_probs = (1 - alpha) * rf_probs + alpha * prior
    rf_probs = rf_probs / rf_probs.sum()

    # Poisson blend (pokud máme λh/λa ve featurech)
    pois = None
    try:
        lh = float(features.get("pois_lambda_home", np.nan))
        la = float(features.get("pois_lambda_away", np.nan))
        if np.isfinite(lh) and np.isfinite(la):
            pp = _poisson_probs_from_lambdas(lh, la, max_goals=12)
            # pořadí podle label_enc.classes_ (např. ['A','D','H'])
            order = list(label_enc.classes_)
            pois_vec = np.array([pp["p_away"], pp["p_draw"], pp["p_home"]])  # A,D,H
            # přeuspořádej do stejného pořadí jako rf_probs
            idx = {lbl: i for i, lbl in enumerate(["A","D","H"])}
            pois = np.array([pois_vec[idx[l]] for l in order])
    except Exception:
        pois = None

    if pois is not None:
        pois = np.clip(pois, 1e-6, 1.0)
        pois = pois / pois.sum()
        probs = (1 - beta) * rf_probs + beta * pois
        probs = probs / probs.sum()
    else:
        probs = rf_probs

    labels = label_enc.inverse_transform(np.arange(len(probs)))
    mapping = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {mapping[lbl]: float(p * 100) for lbl, p in zip(labels, probs)}


def load_over25_model(path: str | Path = DEFAULT_OVER25_MODEL_PATH):
    try:
        data = joblib.load(Path(path))
        return data["model"], data["feature_names"], data.get("label_encoder")
    except Exception:
        logger.warning("Training over/under model because %s is missing", path)
        model, feature_names, label_enc, _, params, _ = train_over25_model(
            n_splits=2, n_iter=1, max_samples=200
        )
        joblib.dump(
            {
                "model": model,
                "feature_names": list(feature_names),
                "label_encoder": label_enc,
                "best_params": params,
            },
            Path(path),
        )
        return model, feature_names, label_enc


def predict_over25_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_OVER25_MODEL_PATH,
    alpha: float = 0.10,   # shrink k empirickému prioru
    beta: float = 0.5,    # váha Poissonu v blendu
) -> float:
    """Vrať pravděpodobnost (0–100), že padne > 2.5 gólu (Over 2.5)."""
    if model_data is None:
        model_data = load_over25_model(model_path)
    model, feature_names, label_enc = model_data

    X = _clip_features(pd.DataFrame([features], columns=feature_names))
    raw = model.predict_proba(X)[0]

    # mapování na pořadí label_enc (0/1)
    if label_enc is not None:
        expected = np.arange(len(label_enc.classes_))
        model_classes = getattr(model, "classes_", expected)
        probs_full = np.zeros(len(expected))
        for p, cls in zip(raw, model_classes):
            probs_full[int(cls)] = p
        classes = label_enc.classes_
        over_idx = list(classes).index("Over 2.5")
        rf_over = probs_full[over_idx]
    else:
        rf_over = raw[1]

    rf_over = float(np.clip(rf_over, 1e-4, 1 - 1e-4))

    # shrink k empirickému prioru
    prior = getattr(model, "prior_", None)
    if prior is not None and label_enc is not None:
        over_idx = list(label_enc.classes_).index("Over 2.5")
        prior_over = float(prior[over_idx])
    else:
        prior_over = 0.5
    rf_over = (1 - alpha) * rf_over + alpha * prior_over

    # Poisson blend (pokud máme λh/λa ve featurech)
    try:
        lh = float(features.get("pois_lambda_home", np.nan))
        la = float(features.get("pois_lambda_away", np.nan))
        if np.isfinite(lh) and np.isfinite(la):
            pp = _poisson_probs_from_lambdas(lh, la, max_goals=12)
            pois_over = float(np.clip(pp["p_over25"], 1e-6, 1 - 1e-6))
            prob = (1 - beta) * rf_over + beta * pois_over
        else:
            prob = rf_over
    except Exception:
        prob = rf_over

    return float(prob * 100)



__all__ = [
    "train_model",
    "train_over25_model",
    "save_model",
    "predict_outcome",
    "construct_features_for_match",
    "predict_proba",
    "load_model",
    "load_over25_model",
    "predict_over25_proba",
]
