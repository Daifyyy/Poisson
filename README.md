# ⚽ Poisson Football Match Predictor

Komplexní aplikace pro odhad výsledků fotbalových zápasů pomocí **Poissonova rozdělení**. Aplikace nabízí přehled lig, detailní statistiky týmů a predikce pro jednotlivé zápasy i hromadné tiketové tipy. Data lze aktualizovat z veřejných CSV zdrojů nebo přes API-Football.

## ✨ Funkce
- Predikce výsledků zápasů (pravděpodobnost 1/X/2, over/under 2.5 gólů, oba týmy skórují)
- Hromadná predikce více zápasů s cachováním výsledků
- Detailní profil týmu včetně ELO ratingu, xG a GII
- Přehled ligy s tabulkou a základními statistikami
- Strength of Schedule (SOS) s průměrnou kvalitou soupeřů podle ELO či xG
- Křížový `team_index` pro srovnání klubů napříč ligami: počítá 0.5× normalizovaný xG rozdíl, 0.5× relativní týmové ELO a 0.1× strength-of-schedule a vše škáluje sílou ligy, takže velkokluby jsou řazeny odpovídajícím způsobem
- Aktualizace datasetů skriptem nebo API
- Jednotkové testy pokrývající klíčové funkce

## 📊 Datové zdroje
Základní data jsou čerpána z [football-data.co.uk](https://www.football-data.co.uk/) a doplňkově lze využít [API-Football](https://www.api-football.com/). V repozitáři jsou již připravena spojena data pro několik evropských lig:

- Premier League (E0)
- Championship (E1)
- Bundesliga (D1) a 2. Bundesliga (D2)
- La Liga (SP1)
- Serie A (I1)
- Ligue 1 (F1)
- Eredivisie (N1)
- Primeira Liga (P1)
- Jupiler League (B1)
- Turkish Super Lig (T1)

Všechny CSV soubory najdeš ve složce `data/`.

Základní CSV musí obsahovat alespoň tyto sloupce:

```
Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS, HST, AST, HC, AC, HY, AY, HR, AR, HF, AF
```

## 🛠️ Instalace

```bash
git clone https://github.com/Daifyyy/Poisson.git
cd Poisson

# doporučeno: vytvoření virtuálního prostředí
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Chceš-li stahovat data přes API-Football, vytvoř soubor `.env` s proměnnou `API_FOOTBALL_KEY`.

## 🚀 Spuštění aplikace

```bash
streamlit run app.py
```

Po spuštění otevři zobrazenou URL v prohlížeči.

## 🔄 Aktualizace dat
- **CSV z football-data.co.uk**: `python scripts/update_league_data.py`
- **API-Football**: `python update_all_leagues_from_api.py` (vyžaduje `API_FOOTBALL_KEY`)
- **League penalty coefficients**: `python update_league_penalty_coefficients.py`

Stažené soubory se ukládají do složky `data/`.

### League penalty coefficients
Tabulka `data/league_penalty_coefficients.csv` uchovává sílu jednotlivých lig,
která se používá při výpočtu křížového `team_indexu`. Aplikace tento soubor
načítá při startu, takže koeficienty jsou konzistentní napříč spuštěními.
Po přidání nové ligy nebo změně dat spusť skript výše a commitni aktualizovaný
CSV, aby se změny propsaly i do aplikace.

## ✅ Testy

Projekt obsahuje sadu jednotkových testů. Pro jejich spuštění použij:

```bash
pytest
```

## 📁 Struktura projektu

```
Poisson/
├── app.py                 # Streamlit aplikace
├── data/                  # Připravená a stažená data
├── scripts/               # Pomocné skripty pro aktualizaci dat
├── sections/              # Oddělené sekce UI
├── tests/                 # Unit testy
├── utils/                 # Logika pro práci s daty a frontendem
├── requirements.txt
└── README.md
```

## 📌 Autor
Vytvořeno s ⚽ a 🧠 od [Daifyyy](https://github.com/Daifyyy)
