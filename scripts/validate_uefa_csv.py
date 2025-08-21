#!/usr/bin/env python3
"""
Validuje CSV soubory se zápasy UEFA pohárů (CL, EL, ECL, combined).

Kontroluje:
- existenci souboru
- nepřítomnost prázdného obsahu
- přítomnost všech požadovaných sloupců

Používá se pro GitHub Actions nebo lokální kontrolu.
"""

import pandas as pd
from pathlib import Path

# Soubory ke kontrole
FILES = {
    "CL": "CL_combined_full.csv",
    "EL": "EL_combined_full.csv",
    "ECL": "ECL_combined_full.csv",
    "ALL": "all_cups_combined.csv",
}

REQUIRED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season", "Competition"
]

DATA_DIR = Path("data")


def validate_csv(name: str, filename: str) -> None:
    path = DATA_DIR / filename
    print(f"🔍 Kontrola souboru: {filename} [{name}]")

    if not path.exists():
        print(f"❌ Soubor neexistuje: {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Chyba při načítání: {e}")
        return

    if df.empty:
        print(f"⚠️  Soubor je prázdný: {filename}")
        return

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ Chybějící sloupce: {missing_cols}")
        return

    # Přehled
    print(f"✅ OK – řádků: {len(df)}, datumy: {df['Date'].min()} → {df['Date'].max()}")


def main():
    for key, file in FILES.items():
        validate_csv(key, file)
        print("")  # prázdný řádek mezi soubory


if __name__ == "__main__":
    main()
