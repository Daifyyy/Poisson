import pandas as pd
import requests
from io import StringIO
from datetime import datetime

SEASONS = {
    "2025/24": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "2024/25": "https://www.football-data.co.uk/mmz4281/2324/E1.csv",
    "2022/23": "https://www.football-data.co.uk/mmz4281/2223/E1.csv",
}

def download_csv(url):
    print(f"⬇️ Stahuji: {url}")
    response = requests.get(url)
    response.encoding = 'utf-8'
    return pd.read_csv(StringIO(response.text))

def parse_dates(df):
    # Oprava a převod sloupce 'Date' na datetime
    for col in df.columns:
        if col.strip().lower() == "date":
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            except Exception as e:
                print(f"❌ Chyba při převodu datumu: {e}")
    return df

def normalize_columns(df):
    # Odstranění whitespace a standardizace velkých písmen
    df.columns = [col.strip() for col in df.columns]
    return df

def main():
    dfs = []

    for season, url in SEASONS.items():
        df = download_csv(url)
        df = normalize_columns(df)
        df = parse_dates(df)
        df["Season"] = season
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values("Date")

    # Uložení
    output_path = "E1_combined_full_updated.csv"
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Soubor byl uložen jako {output_path}")
    print(f"📅 Rozsah dat: {combined_df['Date'].min().date()} až {combined_df['Date'].max().date()}")
    print(f"🔢 Počet zápasů: {len(combined_df)}")

if __name__ == "__main__":
    main()
