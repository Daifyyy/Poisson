#!/usr/bin/env python3
"""
Sloučí CSV soubory pro CL, EL, ECL do jednoho přehledného souboru.

Vstupy:
- data/CL_combined_full.csv
- data/EL_combined_full.csv
- data/ECL_combined_full.csv

Výstup:
- data/all_cups_combined.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
CUPS = {
    "CL": "CL_combined_full.csv",
    "EL": "EL_combined_full.csv",
    "ECL": "ECL_combined_full.csv",
}

def main():
    frames = []
    for cup, filename in CUPS.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"⚠️ Soubor nenalezen: {path}")
            continue

        df = pd.read_csv(path)

        if "Competition" not in df.columns:
            df["Competition"] = cup

        frames.append(df)

    if not frames:
        print("❌ Žádná data nebyla načtena.")
        return

    combined = pd.concat(frames, ignore_index=True)
    output_path = DATA_DIR / "all_cups_combined.csv"
    combined.to_csv(output_path, index=False)
    print(f"✅ Uloženo {len(combined)} řádků do {output_path}")

if __name__ == "__main__":
    main()
