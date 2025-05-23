# import os
# import time
# import requests
# import pandas as pd
# import json
# from dotenv import load_dotenv

# # ✅ Načtení .env a ověření API klíče
# load_dotenv()
# API_KEY = os.getenv("API_FOOTBALL_KEY")
# if not API_KEY:
#     raise EnvironmentError("❌ API_FOOTBALL_KEY nebyl nalezen! Zkontroluj soubor .env a správně nastav klíč.")

# BASE_URL = "https://v3.football.api-sports.io/"
# HEADERS = {"x-apisports-key": API_KEY}

# # ✅ Vytvoř složku data/ pokud neexistuje
# os.makedirs("data", exist_ok=True)

# def get_fixtures_by_league_and_season(league_id, season):
#     url = f"{BASE_URL}fixtures"
#     params = {"league": league_id, "season": season}
#     all_fixtures = []
#     page = 1

#     while True:
#         params["page"] = page
#         response = requests.get(url, headers=HEADERS, params=params)

#         # ⚠️ Kontrola stavového kódu a obsahu odpovědi
#         if response.status_code != 200:
#             print(f"❌ Chyba API (status {response.status_code}): {response.text}")
#             break

#         data = response.json()

#         if not data.get("response"):
#             print(f"⚠️ Prázdná odpověď pro ligu {league_id}, sezónu {season}.")
#             print(json.dumps(data, indent=2, ensure_ascii=False))
#             break

#         all_fixtures.extend(data["response"])

#         if page >= data.get("paging", {}).get("total", 1):
#             break
#         page += 1
#         time.sleep(1)  # prevent hitting rate limits

#     return all_fixtures

# def get_stat(stats, name, is_home=True):
#     team_index = 0 if is_home else 1
#     try:
#         for stat in stats[team_index]["statistics"]:
#             if stat["type"] == name:
#                 return stat["value"] or 0
#     except Exception:
#         return 0
#     return 0

# def map_fixture_to_row(fixture):
#     try:
#         if not fixture.get("fixture") or not fixture["fixture"].get("date"):
#             raise ValueError("chybí datum")
#         if not fixture.get("teams") or not fixture["teams"]["home"].get("name") or not fixture["teams"]["away"].get("name"):
#             raise ValueError("chybí název týmů")

#         stats = fixture.get("statistics", [])
#         return {
#             "Date": fixture["fixture"]["date"][:10],
#             "HomeTeam": fixture["teams"]["home"]["name"],
#             "AwayTeam": fixture["teams"]["away"]["name"],
#             "FTHG": fixture["goals"]["home"],
#             "FTAG": fixture["goals"]["away"],
#             "FTR": "H" if fixture["goals"]["home"] > fixture["goals"]["away"]
#                     else "A" if fixture["goals"]["home"] < fixture["goals"]["away"] else "D",
#             "HTHG": fixture["score"]["halftime"]["home"],
#             "HTAG": fixture["score"]["halftime"]["away"],
#             "HTR": None,
#             "HS": get_stat(stats, "Shots total", True),
#             "AS": get_stat(stats, "Shots total", False),
#             "HST": get_stat(stats, "Shots on goal", True),
#             "AST": get_stat(stats, "Shots on goal", False),
#             "HF": get_stat(stats, "Fouls", True),
#             "AF": get_stat(stats, "Fouls", False),
#             "HC": get_stat(stats, "Corners", True),
#             "AC": get_stat(stats, "Corners", False),
#             "HY": get_stat(stats, "Yellow Cards", True),
#             "AY": get_stat(stats, "Yellow Cards", False),
#             "HR": get_stat(stats, "Red Cards", True),
#             "AR": get_stat(stats, "Red Cards", False),
#         }
#     except Exception as e:
#         print(f"⚠️ Přeskočen zápas kvůli chybě: {e}")
#         return None

# def download_api_data(league_id, seasons, output_csv):
#     all_rows = []
#     for season in seasons:
#         print(f"🔄 Načítám sezónu {season}...")
#         fixtures = get_fixtures_by_league_and_season(league_id, season)
#         if fixtures:
#             print("🔎 Ukázka 1. zápasu:")
#             print(json.dumps(fixtures[0], indent=2, ensure_ascii=False))
#         for fixture in fixtures:
#             mapped = map_fixture_to_row(fixture)
#             if mapped:
#                 all_rows.append(mapped)

#     df = pd.DataFrame(all_rows)
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Uloženo do {output_csv}")

import os
import time
import requests
import pandas as pd
import json
from dotenv import load_dotenv

# ✅ Načtení .env a ověření API klíče
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise EnvironmentError("❌ API_FOOTBALL_KEY nebyl nalezen! Zkontroluj soubor .env a správně nastav klíč.")

BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {"x-apisports-key": API_KEY}

# ✅ Vytvoř složku data/ pokud neexistuje
os.makedirs("data", exist_ok=True)

def get_statistics_for_fixture(fixture_id):
    url = f"{BASE_URL}fixtures/statistics"
    params = {"fixture": fixture_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"❌ Statistika nedostupná pro fixture {fixture_id} (status {response.status_code})")
        return []
    return response.json().get("response", [])

def get_fixtures_by_league_and_season(league_id, season):
    url = f"{BASE_URL}fixtures"
    params = {"league": league_id, "season": season}

    response = requests.get(url, headers=HEADERS, params=params)

    # ⚠️ Kontrola stavového kódu a obsahu odpovědi
    if response.status_code != 200:
        print(f"❌ Chyba API (status {response.status_code}): {response.text}")
        return []

    data = response.json()

    if not data.get("response"):
        print(f"⚠️ Prázdná odpověď pro ligu {league_id}, sezónu {season}.")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return []

    return data["response"]

def get_stat(stats, name, is_home=True):
    team_index = 0 if is_home else 1
    try:
        for stat in stats[team_index]["statistics"]:
            if stat["type"] == name:
                return stat["value"] or 0
    except Exception:
        return 0
    return 0

def map_fixture_to_row(fixture):
    try:
        if not fixture.get("fixture") or not fixture["fixture"].get("date"):
            raise ValueError("chybí datum")
        if not fixture.get("teams") or not fixture["teams"]["home"].get("name") or not fixture["teams"]["away"].get("name"):
            raise ValueError("chybí název týmů")

        fixture_id = fixture["fixture"]["id"]
        stats = get_statistics_for_fixture(fixture_id)

        return {
            "Date": fixture["fixture"]["date"][:10],
            "HomeTeam": fixture["teams"]["home"]["name"],
            "AwayTeam": fixture["teams"]["away"]["name"],
            "FTHG": fixture["goals"]["home"],
            "FTAG": fixture["goals"]["away"],
            "FTR": "H" if fixture["goals"]["home"] > fixture["goals"]["away"]
                    else "A" if fixture["goals"]["home"] < fixture["goals"]["away"] else "D",
            "HTHG": fixture["score"]["halftime"]["home"],
            "HTAG": fixture["score"]["halftime"]["away"],
            "HTR": None,
            "HS": get_stat(stats, "Shots total", True),
            "AS": get_stat(stats, "Shots total", False),
            "HST": get_stat(stats, "Shots on goal", True),
            "AST": get_stat(stats, "Shots on goal", False),
            "HF": get_stat(stats, "Fouls", True),
            "AF": get_stat(stats, "Fouls", False),
            "HC": get_stat(stats, "Corners", True),
            "AC": get_stat(stats, "Corners", False),
            "HY": get_stat(stats, "Yellow Cards", True),
            "AY": get_stat(stats, "Yellow Cards", False),
            "HR": get_stat(stats, "Red Cards", True),
            "AR": get_stat(stats, "Red Cards", False),
        }
    except Exception as e:
        print(f"⚠️ Přeskočen zápas kvůli chybě: {e}")
        return None

def download_api_data(league_id, seasons, output_csv):
    all_rows = []
    for season in seasons:
        print(f"🔄 Načítám sezónu {season}...")
        fixtures = get_fixtures_by_league_and_season(league_id, season)
        if fixtures:
            print("🔎 Ukázka 1. zápasu:")
            print(json.dumps(fixtures[0], indent=2, ensure_ascii=False))
        for fixture in fixtures:
            mapped = map_fixture_to_row(fixture)
            if mapped:
                all_rows.append(mapped)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Uloženo do {output_csv}")
