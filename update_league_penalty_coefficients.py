"""Recalculate league penalty coefficients and save them to disk."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.poisson_utils import calculate_elo_ratings, load_cup_matches
from utils.poisson_utils.cross_league import WORLD_ELO_MEAN


def main() -> None:
    data_dir = ROOT / "data"

    match_frames: list[pd.DataFrame] = []
    team_league_map: dict[str, str] = {}

    for csv in sorted(data_dir.glob("*_combined_full_updated.csv")):
        league_code = csv.name.split("_")[0]
        df = pd.read_csv(csv)
        match_frames.append(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
        teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
        for team in teams:
            team_league_map[team] = league_code

    cup_df = load_cup_matches(team_league_map, data_dir)
    if not cup_df.empty:
        match_frames.append(cup_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])

    all_matches = pd.concat(match_frames, ignore_index=True)
    elo_dict = calculate_elo_ratings(all_matches)
    elo_df = pd.DataFrame(list(elo_dict.items()), columns=["team", "elo"])
    elo_df["league"] = elo_df["team"].map(team_league_map)
    elo_df = elo_df.dropna(subset=["league"])

    rating_rows = (
        elo_df.groupby("league")["elo"].mean().reset_index()
    )
    rating_rows["penalty_coef"] = rating_rows["elo"] / WORLD_ELO_MEAN
    rating_rows = rating_rows.round({"elo": 3, "penalty_coef": 3})

    out_path = data_dir / "league_penalty_coefficients.csv"
    rating_rows.sort_values("league").to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
