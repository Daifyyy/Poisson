from utils.update_from_api import download_api_data, get_fixtures_by_league_and_season, map_fixture_to_row
from dotenv import load_dotenv
import os
import pandas as pd
import datetime

# Načti .env soubor s API klíčem
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

if not API_KEY:
    raise EnvironmentError("❌ API_FOOTBALL_KEY nebyl nalezen! Zkontroluj soubor .env a správně nastav klíč.")

# Mapa lig v API-FOOTBALL: { kód projektu : API ID }
LEAGUES = {
    "E0": 39,     # Premier League
    "E1": 40,     # Championship
    "SP1": 140,   # La Liga
    "B1": 144,    # Jupiler League
    "D1": 78,     # Bundesliga
    "D2": 79,     # 2. Bundesliga
    "I1": 135,    # Serie A
    "F1": 61,     # Ligue 1
    "N1": 88,     # Eredivisie
    "P1": 94,     # Primeira Liga
    "T1": 203     # Turkish Super Lig
}

SEASONS = [2021]
log_entries = []
log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for code, api_id in LEAGUES.items():
    output_path = f"data/{code}_api_combined_full.csv"
    print(f"\n📥 Stahuji ligu {code} ({api_id}) pro sezóny {SEASONS}...")
    
    try:
        # Načtení existujících dat
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            existing_df["Date"] = pd.to_datetime(existing_df["Date"])
        else:
            existing_df = pd.DataFrame()

        new_rows = []
        for season in SEASONS:
            fixtures = get_fixtures_by_league_and_season(api_id, season)
            for fix in fixtures:
                mapped = map_fixture_to_row(fix)
                if "Date" in mapped and "HomeTeam" in mapped and "AwayTeam" in mapped:
                    new_rows.append(mapped)

        new_df = pd.DataFrame(new_rows)
        new_df["Date"] = pd.to_datetime(new_df["Date"])

        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
        else:
            combined = new_df

        combined = combined.sort_values("Date")
        combined.to_csv(output_path, index=False)

        msg = f"✅ {code}: {len(new_df)} nových zápasů ➕ Celkem: {len(combined)} - OK"
        print(msg)
        log_entries.append(msg)

    except Exception as e:
        msg = f"❌ {code}: CHYBA - {str(e)}"
        print(msg)
        log_entries.append(msg)

# Uložení logu do souboru
log_file = f"data/update_api_log.txt"
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n[{log_time}] Aktualizace lig:\n")
    for entry in log_entries:
        f.write(entry + "\n")

print(f"\n📄 Log uložen do: {log_file}")