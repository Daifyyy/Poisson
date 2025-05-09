import pandas as pd
import requests
import os
from io import StringIO

LEAGUES = {
    "E0": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",  # Premier League
    "E1": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",  # Championship
    "SP1": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",  # La Liga
    "B1": "https://www.football-data.co.uk/mmz4281/2425/B1.csv",  # Jupiler League
    "D1": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",  # Bundesliga
    "D2": "https://www.football-data.co.uk/mmz4281/2425/D2.csv",  # 2. Bundesliga    
    "I1": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",  # Serie A
    "F1": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",  # Ligue 1
    "N1": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",  # Eredivisie
    "P1": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",  # Primeira Liga
    "T1": "https://www.football-data.co.uk/mmz4281/2425/T1.csv",  # Super League (Switzerland)
}


def normalize_keys(df):
    """Normalizace klíčových sloupců pro přesné porovnání."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    return df

def update_all_leagues():
    messages = []

    for code, url in LEAGUES.items():
        try:
            print(f"🔄 Stahuji {code}...")
            response = requests.get(url)
            if response.status_code != 200:
                messages.append(f"❌ {code}: Stažení selhalo.")
                continue

            df_new = pd.read_csv(StringIO(response.text))
            df_new = normalize_keys(df_new)

            path = f"data/{code}_combined_full_updated.csv"
            if not os.path.exists(path):
                messages.append(f"❌ {code}: Soubor {path} neexistuje.")
                continue

            df_existing = pd.read_csv(path)
            df_existing = normalize_keys(df_existing)

            # Doplnit chybějící sloupce (např. Div) hodnotou podle ligy
            for col in df_existing.columns:
                if col not in df_new.columns:
                    df_new[col] = code

            # Ořezat a seřadit sloupce podle existujícího souboru
            df_new = df_new[df_existing.columns]

            # Najít nové zápasy
            merge_keys = ["Date", "HomeTeam", "AwayTeam"]
            merged = df_new.merge(df_existing[merge_keys], on=merge_keys, how="left", indicator=True)
            new_rows = df_new[merged["_merge"] == "left_only"]

            if not new_rows.empty:
                df_combined = pd.concat([df_existing, new_rows], ignore_index=True)
                df_combined = df_combined.sort_values("Date")
                # 🛠️ Převést sloupec Date zpět do formátu den/měsíc/rok
                if "Date" in df_combined.columns:
                    df_combined["Date"] = df_combined["Date"].dt.strftime("%d/%m/%Y")
                df_combined.to_csv(path, index=False)
                messages.append(f"✅ {code}: Přidáno {len(new_rows)} nových zápasů.")
            else:
                messages.append(f"ℹ️ {code}: Žádné nové zápasy.")
        except Exception as e:
            messages.append(f"❌ {code}: Chyba – {str(e)}")

    return messages

if __name__ == "__main__":
    logs = update_all_leagues()
    for log in logs:
        print(log)
