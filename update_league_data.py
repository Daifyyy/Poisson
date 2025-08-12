import os
import sys
import time
import io
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

season_code = "2526"
LEAGUE_CODES = ["E0", "E1", "D1", "D2", "SP1", "I1", "F1", "N1", "P1", "B1"]
LEAGUES = {code: f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv" for code in LEAGUE_CODES}

EXPECTED_COLS = [
    "Div","Date","Time","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
    "HTHG","HTAG","HTR","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR",
    "Avg>2.5","Avg<2.5"
]

def log(msg: str):
    print(msg, flush=True)

def flip_scheme(url: str) -> str:
    if url.startswith("https://"):
        return "http://" + url[len("https://"):]
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url

def robust_get(url: str, tries: int = 4, timeout: int = 20) -> requests.Response | None:
    """
    Retry GET. For HTTP 200 always return the response (even if HTML),
    and let the CSV parser decide. This fixes false negatives like the user's logs.
    """
    last_exc = None
    flipped_once = False
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (compatible; PoissonBot/1.1)"
            })
            status = r.status_code
            clen = len(r.content) if r.content is not None else 0
            ctype = r.headers.get("Content-Type", "")
            log(f"   ↳ HTTP {status}, bytes={clen}, ctype='{ctype}' (try {attempt}/{tries})")
            if status == 200 and r.content:
                return r  # trust 200; parser will validate CSV
            if status in (300, 301, 302, 303, 307, 308, 403, 429, 500, 502, 503, 504):
                time.sleep(1.0 * attempt)
                if attempt >= 2 and not flipped_once:
                    new_url = flip_scheme(url)
                    if new_url != url:
                        log(f"   ↳ Flipping scheme and retrying: {new_url}")
                        url = new_url
                        flipped_once = True
                continue
            return None
        except Exception as e:
            last_exc = e
            log(f"   ↳ Exception: {e} (try {attempt}/{tries})")
            time.sleep(1.0 * attempt)
    if last_exc:
        log(f"   ↳ Final exception: {last_exc}")
    return None

def normalize_columns(cols) -> list[str]:
    return (
        pd.Index(cols).astype(str)
        .str.strip()
        .str.replace("﻿", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .tolist()
    )

def _looks_like_csv(text_first_kb: str) -> bool:
    line = next((ln for ln in text_first_kb.splitlines() if ln.strip()), "")
    if not line:
        return False
    # Header hint or comma density
    if "HomeTeam" in line and "AwayTeam" in line:
        return True
    return line.count(",") >= 5

def read_csv_safely(content: bytes, expected_div: str, content_type: str = "") -> pd.DataFrame:
    # Quick HTML rejection (now here, not in network layer)
    head = content[:1024].decode("utf-8", errors="ignore").lower()
    if "<html" in head and "csv" not in content_type.lower():
        log("⚠️ Odpověď vypadá jako HTML, zkusím přesto parsovat jako CSV (někteří hosté vrací HTML s embedded CSV).")

    # Choose encoding
    txt = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            txt = content.decode(enc, errors="ignore")
            encoding_used = enc
            break
        except Exception:
            continue
    if txt is None:
        log("❌ Nepodařilo se dekódovat obsah ani jako utf-8-sig, ani latin-1.")
        return pd.DataFrame()

    # If it doesn't look like CSV at all and Content-Type isn't CSV-ish, abort early
    if not _looks_like_csv(txt[:2048]) and ("csv" not in content_type.lower()):
        log("⚠️ Nevidím CSV hlavičku/čárky – pravděpodobně HTML/mezikrok. Přeskakuji.")
        return pd.DataFrame()

    buf = io.StringIO(txt)
    # Peek header presence
    first_line = next((ln for ln in txt.splitlines() if ln.strip()), "")
    has_header = ("Div" in first_line and "Date" in first_line and "HomeTeam" in first_line and "AwayTeam" in first_line)

    try:
        if has_header:
            df = pd.read_csv(buf, sep=",", engine="python")
            df.columns = normalize_columns(df.columns)
            keep = [c for c in EXPECTED_COLS if c in df.columns]
            df = df[keep] if keep else df
        else:
            buf.seek(0)
            df = pd.read_csv(buf, sep=",", engine="python", header=None,
                             names=EXPECTED_COLS, usecols=range(len(EXPECTED_COLS)))
            log("⚠️ CSV nemělo hlavičku – byla doplněna ručně.")
    except Exception as e:
        log(f"❌ Chyba při čtení CSV: {e}")
        return pd.DataFrame()

    # Filter on expected division
    if "Div" in df.columns:
        df = df[df["Div"].astype(str).str.strip() == expected_div]
    else:
        log("⚠️ Sloupec 'Div' chybí – vracím prázdná data pro bezpečnost.")
        return pd.DataFrame()

    # Date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    # Drop all-NA rows (happens when file has only header)
    df = df.dropna(how="all")
    return df

def generate_key(df: pd.DataFrame) -> pd.Series:
    for col in ("Date", "HomeTeam", "AwayTeam"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' for key generation.")
    return (
        df["Date"].astype(str).str[:10].str.strip() + "|" +
        df["HomeTeam"].astype(str).str.lower().str.strip() + "|" +
        df["AwayTeam"].astype(str).str.lower().str.strip()
    )

def update_league(league_code: str):
    url = LEAGUES[league_code]
    log(f"\n🔄 Aktualizuji ligu {league_code}…")
    log(f"   URL: {url}")

    resp = robust_get(url)
    if not resp:
        log(f"⚠️ Přeskočeno: {league_code} – nepodařilo se získat platný CSV obsah.")
        return

    df_new = read_csv_safely(resp.content, expected_div=league_code, content_type=resp.headers.get("Content-Type",""))
    if df_new.empty:
        log(f"⚠️ Přeskočeno: {league_code} – žádná data po načtení.")
        return

    log(f"🧪 {league_code}: sloupce={df_new.columns.tolist()} | řádků={len(df_new)}")

    updated_path = os.path.join(DATA_DIR, f"{league_code}_combined_full_updated.csv")
    if os.path.exists(updated_path):
        try:
            df_existing = pd.read_csv(updated_path)
        except Exception as e:
            log(f"⚠️ Problém při čtení existujícího souboru: {e}. Pokusím se pokračovat s prázdným.")
            df_existing = pd.DataFrame(columns=df_new.columns)
    else:
        df_existing = pd.DataFrame(columns=df_new.columns)

    df_existing.columns = [str(c).strip() for c in df_existing.columns]
    keep = [c for c in df_existing.columns if c in df_new.columns] or list(df_new.columns)
    df_new = df_new[keep]
    df_existing = df_existing.reindex(columns=keep)

    try:
        df_new["match_key"] = generate_key(df_new)
        df_existing["match_key"] = generate_key(df_existing) if not df_existing.empty else pd.Series(dtype=str)
    except KeyError as e:
        log(f"⚠️ Přeskočeno: {league_code} – chybí klíčové sloupce: {e}")
        return

    df_new_rows = df_new[~df_new["match_key"].isin(df_existing.get("match_key", pd.Series(dtype=str)))].drop(columns="match_key")
    df_existing = df_existing.drop(columns="match_key", errors="ignore")

    if not df_new_rows.empty:
        df_combined = pd.concat([df_existing, df_new_rows], ignore_index=True)
        if "Date" in df_combined.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_combined["Date"]):
                df_combined["Date"] = pd.to_datetime(df_combined["Date"], errors="coerce")
            df_combined = df_combined.sort_values("Date")

        df_combined = df_combined.loc[:, ~pd.Index(df_combined.columns).str.contains(r"^Unnamed")]
        if "Date" in df_combined.columns and pd.api.types.is_datetime64_any_dtype(df_combined["Date"]):
            df_combined["Date"] = df_combined["Date"].dt.strftime("%Y-%m-%d")

        df_combined.to_csv(updated_path, index=False)
        log(f"✅ Přidáno {len(df_new_rows)} nových zápasů do {updated_path}")
    else:
        log("ℹ️ Žádné nové zápasy – soubor je aktuální.")

def update_all_leagues():
    for code in LEAGUES:
        update_league(code)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        for c in args:
            if c in LEAGUES:
                update_league(c)
            else:
                log(f"⚠️ Neznámý kód ligy: {c}")
    else:
        update_all_leagues()
