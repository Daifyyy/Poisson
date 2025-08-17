"""Recalculate league penalty coefficients and save them to disk."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.poisson_utils import calculate_elo_ratings
from utils.poisson_utils.cross_league import WORLD_ELO_MEAN


def main() -> None:
    data_dir = ROOT / "data"
    rating_rows = []
    for csv in sorted(data_dir.glob("*_combined_full_updated.csv")):
        league_code = csv.name.split("_")[0]
        df = pd.read_csv(csv)
        elo_dict = calculate_elo_ratings(df)
        avg_elo = sum(elo_dict.values()) / len(elo_dict)
        rating_rows.append(
            {
                "league": league_code,
                "elo": round(avg_elo, 3),
                "penalty_coef": round(avg_elo / WORLD_ELO_MEAN, 3),
            }
        )
    out_path = data_dir / "league_penalty_coefficients.csv"
    pd.DataFrame(rating_rows).sort_values("league").to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
