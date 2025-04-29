# import os
# import pandas as pd
# import requests
# from io import StringIO

# # Liga -> (kód, URL)
# LEAGUES = {
#     "E0": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
#     # "E1": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
#     # "D1": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
#     # "D2": "https://www.football-data.co.uk/mmz4281/2425/D2.csv",
#     # "SP1": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
#     # "I1": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
#     # "F1": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
#     # "N1": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
#     # "P1": "https://www.football-data.co.uk/mmz4281/2425/P1.csv"
# }

# def update_league(league_code):
#     print(f"\n🔄 Aktualizuji ligu {league_code}...")

#     url = LEAGUES[league_code]
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"❌ Nelze stáhnout {url}")
#         return

#     df_new = pd.read_csv(StringIO(response.text))
#     df_new['Date'] = pd.to_datetime(df_new['Date'], dayfirst=True, errors='coerce')

#     updated_path = f"data/{league_code}_combined_full_updated.csv"
#         # Pokud existuje aktualizovaný soubor, načti ho
#     if os.path.exists(updated_path):
#         df_existing = pd.read_csv(updated_path)
#         df_existing['Date'] = pd.to_datetime(df_existing['Date'], dayfirst=True, errors='coerce')
        
#         # Omez df_new pouze na existující sloupce
#         df_new = df_new[[col for col in df_existing.columns if col in df_new.columns]]
#     else:
#         df_existing = pd.DataFrame(columns=df_new.columns)


#     merge_cols = ["Date", "HomeTeam", "AwayTeam"]
#     df_merge = df_new.merge(df_existing[merge_cols], on=merge_cols, how="left", indicator=True)
#     df_new_rows = df_merge[df_merge["_merge"] == "left_only"].drop(columns="_merge")

#     if not df_new_rows.empty:
#         df_combined = pd.concat([df_existing, df_new_rows], ignore_index=True)
#         df_combined = df_combined.sort_values("Date")
#         df_combined.to_csv(updated_path, index=False)
#         print(f"✅ Přidáno {len(df_new_rows)} nových zápasů do {updated_path}")
#     else:
#         print("ℹ️ Žádné nové zápasy – soubor je aktuální.")

# def update_all_leagues():
#     for code in LEAGUES:
#         update_league(code)

# # 🔁 Spustit automaticky při spuštění souboru
# if __name__ == "__main__":
#     update_all_leagues()
