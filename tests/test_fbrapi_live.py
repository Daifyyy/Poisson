import os
import time
import json
from pathlib import Path

import requests
import pandas as pd

BASE = "https://fbrapi.com"
LEAGUE_ID = 9
SEASON_ID = "2024-2025"  # případně "2023-2024"

SLEEP = 3.3  # rate-limit ~1 req / 3s


def headers():
    api_key = os.getenv("FBRAPI_KEY", "").strip()
    if not api_key:
        raise SystemExit("❌ Chybí proměnná prostředí FBRAPI_KEY.")
    return {"X-API-Key": api_key, "User-Agent": "Poisson-FBR/1.0"}


def get(path, params=None):
    url = f"{BASE}{path}"
    r = requests.get(url, params=params or {}, headers=headers(), timeout=60)
    if r.status_code in (401, 403):
        raise SystemExit(f"❌ Auth chyba {r.status_code}: {r.text[:200]}")
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()


def main():
    outdir = Path("data_fbrapi")
    outdir.mkdir(exist_ok=True)

    print("✅ Test 1/4 – /teams …")
    teams_json = get("/teams", {"league_id": LEAGUE_ID, "season_id": SEASON_ID})
    teams = pd.DataFrame(teams_json.get("data", []))
    if teams.empty:
        raise SystemExit("❌ /teams vrátil prázdná data.")
    print(teams.head(5))
    teams.to_csv(outdir / f"teams_{LEAGUE_ID}_{SEASON_ID}.csv", index=False)

    # zkusíme najít Arsenal (nebo vezmeme první tým)
    preferred = teams[teams["team"].str.contains("Arsenal", case=False, na=False)]
    team_row = preferred.iloc[0] if not preferred.empty else teams.iloc[0]
    team_id = int(team_row["team_id"])
    team_name = team_row["team"]
    print(f"➡️  Zvolen tým: {team_name} (team_id={team_id})")

    print("✅ Test 2/4 – /matches …")
    matches_json = get("/matches", {"league_id": LEAGUE_ID, "season_id": SEASON_ID})
    matches = pd.DataFrame(matches_json.get("data", []))
    print(matches[["match_id", "date"]].head(5))
    matches.to_csv(outdir / f"matches_{LEAGUE_ID}_{SEASON_ID}.csv", index=False)

    print("✅ Test 3/4 – /team-match-stats …")
    tstats_json = get(
        "/team-match-stats",
        {"team_id": team_id, "league_id": LEAGUE_ID, "season_id": SEASON_ID},
    )
    tstats = pd.DataFrame(tstats_json.get("data", []))
    if tstats.empty:
        raise SystemExit("❌ /team-match-stats vrátil prázdná data.")
    print(tstats.head(10))
    tstats.to_csv(
        outdir / f"teamstats_{team_id}_{LEAGUE_ID}_{SEASON_ID}.csv", index=False
    )

    # Volitelné: rychlý souhrn xG/xGA na zápas pro zvolený tým
    if {"xg", "xga"}.issubset(tstats.columns):
        xg_per_match = tstats["xg"].astype(float).mean()
        xga_per_match = tstats["xga"].astype(float).mean()
        print(
            f"📊 {team_name} – průměr xG/zápas: {xg_per_match:.2f}, xGA/zápas: {xga_per_match:.2f}"
        )

    print("✅ Hotovo. CSV uložena do:", outdir.resolve())


if __name__ == "__main__":
    main()
