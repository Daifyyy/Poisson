"""Train and save Random Forest models for outcome and over/under 2.5 goals."""

import argparse

from utils.ml.random_forest import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OVER25_MODEL_PATH,
    save_model,
    train_model,
    train_over25_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--recent-years", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    model, features, le, score, params, metrics = train_model(
        args.data_dir,
        n_splits=args.n_splits,
        recent_years=args.recent_years,
        n_iter=args.n_iter,
        max_samples=args.max_samples,
    )
    save_model(model, features, le, path=DEFAULT_MODEL_PATH, best_params=params)
    print(f"Outcome model trained with score {score:.3f} and saved to {DEFAULT_MODEL_PATH}")

    o25_model, o25_features, o25_le, o25_score, o25_params, o25_metrics = train_over25_model(
        args.data_dir,
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
        f"Over/Under 2.5 model trained with score {o25_score:.3f} and saved to {DEFAULT_OVER25_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
