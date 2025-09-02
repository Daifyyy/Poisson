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

## xG Data Sources

Expected goals values are fetched from multiple providers with the following
priority:

1. **Understat** – primary source
2. **FBR API** – fallback if Understat is unavailable
3. **FBref** – final fallback when the API fails
4. **pseudo‑xG** – computed from match statistics when no external provider
   succeeds

Results are cached to JSON files in `utils/poisson_utils/xg_sources/`:

- `xg_cache.json` stores the final aggregated results
- `understat_xg_cache.json`, `fbrapi_xg_cache.json` and `fbref_xg_cache.json`
  keep provider‑specific responses

FBR API requests require an API key provided via the environment variable
`FBRAPI_KEY`.

Caches persist between runs. Delete the files to refresh the stored data or
allow the scripts to overwrite them when new values are available.

Data from Understat and FBref/StatsBomb is subject to their respective terms of
use. When distributing or displaying the data, provide appropriate attribution
to the original sources.

### Extending Providers

Additional providers can be added by creating a new module in
`utils/poisson_utils/xg_sources/` that implements a
`get_team_xg_xga(team, season, ...) -> Dict[str, float]` function. Include the
module name in the provider chain within `get_team_xg_xga` to control lookup
priority.

### Example

```python
from utils.poisson_utils.xg_sources import get_team_xg_xga

stats = get_team_xg_xga("Arsenal", "2023-2024", league_df=df)
print(stats["xg"], stats["xga"], stats["source"])  # includes provider name
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

### Probability shrinkage

Predikce lze zjemnit tzv. shrinkage faktorem `alpha`, který mísí výstupy
modelu s neutrálním prior.

- Proměnná prostředí `PROBA_ALPHA` (výchozí `0.05`) se načítá v aplikaci a
  předává se do všech volání modelu.
- Hodnoty blíže nule nechávají pravděpodobnosti téměř beze změny, vyšší čísla
  je posouvají ke 33 %/33 %/33 % u výsledku zápasu nebo 50 %/50 % u over/under
  2.5.
- Knihovní funkce `predict_outcome`, `predict_proba` a `predict_over25_proba`
  parametr `alpha` také přijímají, takže jej lze nastavovat i mimo
  Streamlit aplikaci.

## 🔄 Aktualizace dat
- **CSV z football-data.co.uk**: `python scripts/update_league_data.py`
- **API-Football**: `python update_all_leagues_from_api.py` (vyžaduje `API_FOOTBALL_KEY`)
- **Pohárové soutěže z FBref**: `python scripts/update_cup_data_fbref.py`
- **League penalty coefficients**: `python update_league_penalty_coefficients.py`

Stažené soubory se ukládají do složky `data/`.

### League penalty coefficients
Tabulka `data/league_penalty_coefficients.csv` uchovává sílu jednotlivých lig,
která se používá při výpočtu křížového `team_indexu`. Aplikace tento soubor
načítá při startu, takže koeficienty jsou konzistentní napříč spuštěními.
Po přidání nové ligy nebo změně dat spusť skript výše a commitni aktualizovaný
CSV, aby se změny propsaly i do aplikace.

## 📈 Trénování modelu
Skript `scripts/train_models.py` umožňuje trénovat a ladit Random Forest modely.
Trénink využívá chronologické dělení `TimeSeriesSplit`, vyvážené váhy tříd a
po trénování je model obalen `CalibratedClassifierCV` s isotonic regresí.
Hyperparametry se hledají pomocí `RandomizedSearchCV` optimalizovaného na
`log_loss`. Skript po dokončení vypíše také Brierovy skóre a kalibrační křivky
pro jednotlivé třídy. Parametry křížové validace i rozsah vyhledávání
hyperparametrů lze upravit pomocí argumentů příkazové řádky:
=======
Parametry křížové validace i rozsah vyhledávání hyperparametrů lze upravit
pomocí argumentů příkazové řádky:

```bash
python scripts/train_models.py --n-iter 20 --n-splits 5 --recent-years 2
```

Volitelný argument `--max-samples` může omezit počet zpracovaných zápasů pro
rychlé experimenty.

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
