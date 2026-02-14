from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

# On-disk cache for team xG/xGA results
CACHE_FILE = Path(__file__).with_name("fbrapi_xg_cache.json")

# Directory for storing raw FBR API responses for debugging/inspection
RAW_OUTPUT_DIR = Path(__file__).with_name("fbrapi_raw")
RAW_OUTPUT_DIR.mkdir(exist_ok=True)

# Location where a locally stored API key may reside
API_KEY_FILE = Path.home() / ".fbrapi_api_key"

# Project fallback API key (leave empty; prefer env/file)
DEFAULT_API_KEY = ""  # do NOT commit real keys

def _load_cache() -> Dict[str, Dict[str, float]]:
    """Načti JSON cache xG/xGA; při chybě vrať prázdný dict."""
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # typová jistota
            if isinstance(data, dict):
                return data  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass
    return {}

def _save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    """Ulož JSON cache xG/xGA atomicky."""
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(CACHE_FILE)

def _get_api_key() -> Optional[str]:
    """Získej API klíč z env, souboru, nebo fallbacku (pokud je nastaven)."""
    # 1) Proměnná prostředí má nejvyšší prioritu
    key = os.getenv("FBRAPI_KEY")
    if key:
        return key.strip()

    # 2) Lokální soubor v $HOME
    if API_KEY_FILE.exists():
        try:
            content = API_KEY_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
        except OSError:
            pass

    # 3) Fallback konstanta (měla by zůstat prázdná)
    return DEFAULT_API_KEY or None
