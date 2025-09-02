"""Train and save Random Forest models for outcome and over/under 2.5 goals."""

import argparse

from fbrapi_dataset import build_three_seasons
from utils.ml.random_forest import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OVER25_MODEL_PATH,
    save_model,
    train_model,
    train_over25_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest models")
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--seasons", nargs="+", required=True, help="Season identifiers from FBR API")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--recent-years", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    df = build_three_seasons(args.league_id, args.seasons)

    model, features, le, score, params, metrics = train_model(
        df,
        n_splits=args.n_splits,
        recent_years=args.recent_years,
        n_iter=args.n_iter,
        max_samples=args.max_samples,
    )
    save_model(model, features, le, path=DEFAULT_MODEL_PATH, best_params=params)
    print(f"Outcome model trained with log-loss {score:.3f} and saved to {DEFAULT_MODEL_PATH}")
    per_class = metrics.get("per_class", metrics)
    for lbl, m in per_class.items():
        print(
            f"  {lbl}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, brier={m['brier']:.3f}, ece={m['ece']:.3f}"
        )
    print("Baselines:")
    if 'baselines' in metrics:
        print(f"  Frequency log-loss={metrics['baselines']['frequency_log_loss']:.3f}")
        book_ll = metrics['baselines']['bookmaker_log_loss']
        if book_ll is not None:
            print(f"  Bookmaker log-loss={book_ll:.3f}")


    o25_model, o25_features, o25_le, o25_score, o25_params, o25_metrics = train_over25_model(
        df,
        n_splits=args.n_splits,
        recent_years=args.recent_years,
        n_iter=args.n_iter,
        max_samples=args.max_samples,
    )
    save_model(
        o25_model,
        o25_features,
        o25_le,
        path=DEFAULT_OVER25_MODEL_PATH,
        best_params=o25_params,
    )
    print(
        f"Over/Under 2.5 model trained with log-loss {o25_score:.3f} and saved to {DEFAULT_OVER25_MODEL_PATH}"
    )
    for lbl, m in o25_metrics["per_class"].items():
        print(
            f"  {lbl}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, brier={m['brier']:.3f}, ece={m['ece']:.3f}"
        )
    print("Baselines:")
    print(f"  Frequency log-loss={o25_metrics['baselines']['frequency_log_loss']:.3f}")
    book_ll = o25_metrics['baselines']['bookmaker_log_loss']
    if book_ll is not None:
        print(f"  Bookmaker log-loss={book_ll:.3f}")


if __name__ == "__main__":
    main()
