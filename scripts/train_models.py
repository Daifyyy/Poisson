"""Train and save Random Forest models for outcome and over/under 2.5 goals."""

from utils.ml.random_forest import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OVER25_MODEL_PATH,
    save_model,
    train_model,
    train_over25_model,
)


def main(data_dir: str = "data") -> None:
    model, features, le, score, params, metrics = train_model(
        data_dir, n_splits=3, recent_years=1, n_iter=1, max_samples=500
    )
    save_model(model, features, le, path=DEFAULT_MODEL_PATH, best_params=params)
    print(f"Outcome model trained with score {score:.3f} and saved to {DEFAULT_MODEL_PATH}")

    o25_model, o25_features, o25_le, o25_score, o25_params, o25_metrics = train_over25_model(
        data_dir, n_splits=3, recent_years=1, n_iter=1, max_samples=500
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
