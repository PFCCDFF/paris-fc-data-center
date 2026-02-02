import os
import re
import unicodedata
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from pfc_dc.config import BAD_TOKENS


def normalize_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split()).lower()
    return s


def find_local_file_by_normalized_name(folder: str, target_name: str) -> Optional[str]:
    if not os.path.exists(folder):
        return None
    target_norm = normalize_str(target_name)
    for fn in os.listdir(folder):
        if normalize_str(fn) == target_norm:
            return os.path.join(folder, fn)
    return None


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int_numeric_only(df: pd.DataFrame, round_first: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        if round_first:
            out[num_cols] = out[num_cols].round()
        out[num_cols] = out[num_cols].fillna(0)
        out[num_cols] = out[num_cols].astype(int)
    return out


def nettoyer_nom_joueuse(nom):
    if not isinstance(nom, str):
        nom = str(nom) if nom is not None else ""
    s = nom.strip().upper()
    s = (
        s.replace("É", "E").replace("È", "E").replace("Ê", "E")
        .replace("À", "A").replace("Ù", "U")
        .replace("Î", "I").replace("Ï", "I")
        .replace("Ô", "O").replace("Ö", "O")
        .replace("Â", "A").replace("Ä", "A")
        .replace("Ç", "C")
    )
    s = " ".join(s.split())
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    if len(parts) > 1 and parts[0] == parts[1]:
        return parts[0]
    return s


def nettoyer_nom_equipe(nom: str) -> str:
    if nom is None:
        return ""
    s = str(nom).strip().upper()
    s = (
        s.replace("É", "E").replace("È", "E").replace("Ê", "E")
        .replace("À", "A").replace("Ù", "U")
        .replace("Î", "I").replace("Ï", "I")
        .replace("Ô", "O").replace("Ö", "O")
        .replace("Â", "A").replace("Ä", "A")
        .replace("Ç", "C")
    )
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        s = parts[0] if parts else s
    s = " ".join(s.split())
    return s


def looks_like_player(name: str) -> bool:
    n = nettoyer_nom_joueuse(str(name)) if name is not None else ""
    if not n or n in {"NAN", "NONE", "NULL"}:
        return False
    if any(tok in n for tok in BAD_TOKENS):
        return False
    if len(n) <= 2:
        return False
    if re.search(r"\d", n):
        return False
    return True


def split_if_comma(cell: str) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.upper() in {"NAN", "NONE", "NULL"}:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if len(parts) > 1 else [s]


def parse_date_from_gf1_filename(fn: str) -> Optional[datetime]:
    base = os.path.basename(fn)
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{2,4})", base)
    if not m:
        return None
    d, mo, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = "20" + y
    try:
        return datetime(int(y), int(mo), int(d))
    except Exception:
        return None


def extract_season_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    s = str(filename)
    candidates = re.findall(r"\b\d{4}\b", s)
    for c in candidates:
        if c in {"2425", "2526"}:
            return c
    m = re.search(r"(2425|2526)", s)
    return m.group(1) if m else None
