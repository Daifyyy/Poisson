# ⚽ Poisson Football Match Predictor

Tento projekt umožňuje predikci výsledků fotbalových zápasů pomocí **Poissonova rozdělení**. Uživatel si vybere ligu, prohlédne týmové statistiky a zvolí dva týmy, pro které se vygeneruje predikce:

- Pravděpodobnost výhry domácích, remízy a výhry hostů
- Over / Under 2.5 gólů
- Oba týmy skórují (BTTS)

## 📊 Použité datové zdroje
Data obsahují zápasy za posledních 5 let z:
- **Premier League (E0)**
- **La Liga (SP1)**

## 🚀 Spuštění aplikace

### 1. Naklonuj repozitář

```bash
git clone https://github.com/Daifyyy/Poisson.git
cd Poisson
```

### 2. Vytvoř virtuální prostředí (doporučeno)

```bash
python -m venv venv
source venv/bin/activate  # na Windows: venv\Scripts\activate
```

### 3. Nainstaluj závislosti

```bash
pip install -r requirements.txt
```

### 4. Spusť aplikaci

```bash
streamlit run app.py
```

## 📁 Struktura projektu

```
Poisson/
├── data/
│   ├── E0_combined_full.csv
│   └── SP1_combined_full.csv
├── utils/
│   └── poisson_utils.py
├── app.py
├── requirements.txt
└── README.md
```

## 📌 Autor
Vytvořeno s ⚽ a 🧠 od [Daifyyy](https://github.com/Daifyyy)
