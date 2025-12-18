import os
import re
import ast
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
#  AYARLAR (backtest3 + backtest4 birleşik, v5 + PatternScore + HitRate filtresi)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Kolon isimleri ---
COL_SYMBOL = "Sembol"
COL_MAX_5 = "Max_Getiri_%"
COL_MAX_15 = "Max_Getiri_15Gun_%"
COL_ANALIZ_TARIHI = "Analiz_Tarihi_str"
COL_FIYAT_ANALIZ = "Alis_Fiyati_AnalizGunu"   # backtest3'ten geliyor
COL_FIYAT_SON = "Fiyat (Son)"                 # multim/backtest4 tarafında

# --- Risk/Skor ayarları (backtest4'ten) ---
ALPHA_5D_LOW = 0.7
GAMMA_5D_HIGH = 0.2
BETA_15D_LOW = 0.7

THRESH_5D_MIN = 10.0   # 5 gün ortalama min %10
THRESH_15D_MIN = 20.0  # 15 gün ortalama min %20

MIN_N_5D_LOW = 50
MIN_N_5D_HIGH = 30
MIN_N_15D_LOW = 50

MAX_NEG_RATIO_5D_LOW = 0.45
MAX_NEG_RATIO_15D_LOW = 0.40

# --- PatternScore için ayarlar ---
HIT_THR_5 = 20.0    # 5g için başarılı trade eşiği (%)
HIT_THR_15 = 30.0   # 15g için başarılı trade eşiği (%)
MIN_SYMBOLS_5 = 10  # 5g pattern'inde min sembol sayısı (yumuşak koşul, skorla cezalanıyor)
MIN_SYMBOLS_15 = 10

# --- HitRate bazlı pattern filtresi için ek eşikler ---
# NOT: Mevcut veri setinde HitRate5 değerleri genelde 0.1–0.2 aralığında olduğu için
# 5g düşük risk seçiminde HitRate filtresini SERT kullanmıyoruz, sadece sıralamada kullanıyoruz.
MIN_HITRATE5 = 0.60     # Şimdilik sadece sıralama için referans; filtreden çıkarıldı
MIN_SYMBOLS5 = 5        # en az 5 farklı sembolde görülmüş olsun
MIN_N5_HITFILTER = 20   # toplam en az 20 örnek

# --- Çıktı isimleri (v5) ---
BEST_5D_LOW_TEMPLATE = "BEST_COMBOS5_5D_LOW_{date}.csv"
BEST_5D_HIGH_TEMPLATE = "BEST_COMBOS5_5D_HIGH_{date}.csv"
BEST_15D_LOW_TEMPLATE = "BEST_COMBOS5_15D_LOW_{date}.csv"

LIST_1_TEMPLATE = "LISTE5_1_5GUN_DUSUK_RISK_MIN10_{date}.csv"
LIST_2_TEMPLATE = "LISTE5_2_5GUN_YUKSEK_RISK_TOP20_{date}.csv"
LIST_3_TEMPLATE = "LISTE5_3_15GUN_DUSUK_RISK_MIN20_{date}.csv"

COMBINED_RISKY_CSV_TEMPLATE = "RISKLI5_5GUN_VE_15GUN_BIRLESIK_{date}.csv"
COMBINED_RISKY_XLSX_TEMPLATE = "RISKLI5_5GUN_VE_15GUN_BIRLESIK_{date}.xlsx"

# --- v3 backtest + mining default config ---
DEFAULT_CONFIG = {
    "dir": BASE_DIR,
    "force_latest_by_date": True,
    "top_quantile": 0.10,
    "top_quantile_mid": 0.10,
    "mine_combos": True,
    "select_from_latest": False,
    "select_top_n": 200,
    "select_top_n_mid": 200,
    "combo_min_n_single": 120,
    "combo_min_n_pair": 150,
    "combo_topk_single": 30,
    "combo_topk_pair": 50,
    "num_bins": 3,
    "max_cat_unique": 20,
    "latest_source": "file",
    "export_xlsx": True,
    "verbose": False,
    "last_n_files": 0,
    "start_date": None,
    "end_date": None,
    "mid_horizon_days": 15,
}

# ============================================================
#  YARDIMCI FONKSIYONLAR (backtest3'ten aynen)
# ============================================================

_DATE_RE_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def detect_date_from_filename_str(fname: str) -> Optional[pd.Timestamp]:
    m = _DATE_RE_ISO.search(fname)
    if m:
        try:
            return pd.to_datetime(m.group(0))
        except Exception:
            return None
    return None


def _norm_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = (
        s.replace("ı", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
    )
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


_SYMBOL_ALIASES = [
    "sembol",
    "kod",
    "hisse",
    "symbol",
    "ticker",
    "hissekodu",
    "sembulkodu",
    "sembulkod",
]
_PRICE_ALIASES = [
    "fiyatson",
    "fiyat_son",
    "sonfiyat",
    "kapanis",
    "kapanisfiyat",
    "close",
    "price",
    "fiyat",
    "son",
    "sondeger",
    "sondeg",
    "fiyattl",
    "sonfiyattl",
    "fiyat (son)",
]


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {_norm_str(c): c for c in cols}
    cand_norms = {_norm_str(a) for a in candidates}
    for cand in candidates:
        key = _norm_str(cand)
        if key in norm_map:
            return norm_map[key]
    for c in cols:
        if _norm_str(c) in cand_norms:
            return c
    return None


def resolve_symbol_and_price_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    df.columns = [str(c).strip() for c in df.columns]
    sym = resolve_column(df, _SYMBOL_ALIASES + ["Sembol"])
    prc = resolve_column(df, _PRICE_ALIASES)
    return sym, prc


def smart_to_numeric(series: pd.Series) -> pd.Series:
    """Metin/gürültülü sayısal seriyi olabildiğince sayıya çevir."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype("string").str.strip()
    s = s.mask(s.str.len() == 0, pd.NA)
    lower = s.str.lower()
    s = s.mask(lower.isin(["nan", "none", "null", "na", "inf", "-inf"]), pd.NA)
    s = s.str.replace(r"[^0-9,.\-]", "", regex=True)

    sample = s.dropna().head(200)
    try:
        comma_decimal = int(sample.str.contains(r",\d{1,3}$", na=False).sum())
        dot_decimal = int(sample.str.contains(r"\.\d{1,3}$", na=False).sum())
    except Exception:
        comma_decimal = dot_decimal = 0

    if comma_decimal > dot_decimal:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def _read_csv_any(path: str, on_bad_lines="skip") -> Optional[pd.DataFrame]:
    """Farklı encoding denemeleriyle CSV oku."""
    for enc in ["utf-8-sig", "utf-16", "latin1", "cp1254", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines=on_bad_lines)
            if df is not None:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            continue
    return None


def select_multim4_base_files(
    folder: str,
    start_date: Optional[str],
    end_date: Optional[str],
    last_n_files: Optional[int],
) -> Tuple[List[str], List[Tuple[str, Optional[pd.Timestamp]]]]:
    """
    multim4_YYYY-MM-DD.csv dosyalarını tarih bilgisiyle birlikte seçer.
    Tarih filtreleri ve son N dosya seçimi uygular.
    """
    all_csv = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    base = [f for f in all_csv if re.match(r"^multim4_\d{4}-\d{2}-\d{2}\.csv$", f)]
    f_with_dates: List[Tuple[str, Optional[pd.Timestamp]]] = []

    sd = pd.to_datetime(start_date) if start_date else None
    ed = pd.to_datetime(end_date) if end_date else None

    for f in base:
        d = detect_date_from_filename_str(f)
        if sd and d and d < sd:
            continue
        if ed and d and d > ed:
            continue
        f_with_dates.append((f, d))

    f_with_dates.sort(key=lambda x: (x[1] is None, x[1], x[0]))

    if last_n_files and last_n_files > 0:
        f_with_dates = f_with_dates[-last_n_files:]

    return [f for f, _ in f_with_dates], f_with_dates


def classify_columns(df_all: pd.DataFrame, ignore: set, max_cat_unique: int):
    """
    Mining'de kullanılacak kolonları sınıflandır.
    ÖNEMLİ: ignore set'ine performans/etiket kolonları da ekleniyor (v5 farkı).
    """
    numeric: List[str] = []
    binary: List[str] = []
    categorical: List[str] = []

    for c in df_all.columns:
        if c in ignore:
            continue
        s = df_all[c]
        s_num = pd.to_numeric(s, errors="coerce")
        ratio = s_num.notna().sum() / max(len(s_num), 1)
        unique_non_na = set(pd.Series(s.dropna().unique()).tolist())
        is_bin = len(unique_non_na) <= 2 and unique_non_na.issubset(
            {0, 1, "0", "1", 0.0, 1.0}
        )
        if is_bin:
            binary.append(c)
            continue
        if ratio >= 0.8:
            numeric.append(c)
            continue
        if s.dropna().nunique() <= max_cat_unique:
            categorical.append(c)

    return numeric, binary, categorical


def build_rule_mask(df: pd.DataFrame, conditions: List[Dict]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        col = cond["col"]
        op = cond["op"]
        val = cond["val"]
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if op == "==":
            mask &= (df[col] == val)
        elif op == "in_bin":
            x = pd.to_numeric(df[col], errors="coerce")
            left = val.get("left", -np.inf)
            right = val.get("right", np.inf)
            closed = val.get("closed", "right")
            if closed == "right":
                mask &= (x > left) & (x <= right)
            else:
                mask &= (x >= left) & (x < right)
        else:
            mask &= False
    return mask


def compute_metrics(
    y: pd.Series, mask: pd.Series, top_thr: float, base_top_rate: float
) -> Optional[Dict]:
    n = int(mask.sum())
    if n == 0:
        return None
    ret_mean = float(y[mask].mean())
    top_rate = float((y[mask] >= top_thr).mean())
    lift = float(top_rate - base_top_rate)
    return {"N": n, "Ret_Mean": ret_mean, "TopRate": top_rate, "Lift": lift}


def mine_combos(
    df_all: pd.DataFrame,
    y_col: str = "Max_Getiri_%",
    top_quantile: float = 0.2,
    binary_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    num_bins: int = 3,
    min_n_single: int = 200,
    min_n_pair: int = 200,
    topk_single: int = 30,
    topk_pair: int = 50,
) -> pd.DataFrame:
    """
    backtest3'ün combo madenciliği, v5'te de aynen kullanılıyor.
    Sadece ignore set'ini genişleterek performans kolonlarını feature dışı bırakıyoruz.
    """
    y = pd.to_numeric(df_all[y_col], errors="coerce")
    top_thr = y.quantile(1.0 - top_quantile)
    base_top_rate = (y >= top_thr).mean()

    singles: List[Dict] = []

    # Binary
    for c in (binary_cols or []):
        m = (df_all[c] == 1)
        met = compute_metrics(y, m, top_thr, base_top_rate)
        if met and met["N"] >= min_n_single:
            singles.append(
                {
                    "rule_type": "binary",
                    "conditions": [{"col": c, "op": "==", "val": 1}],
                    **met,
                }
            )

    # Categorical
    for c in (categorical_cols or []):
        s = df_all[c]
        vc = s.value_counts(dropna=True)
        for val, cnt in vc.items():
            if cnt < min_n_single:
                continue
            m = (s == val)
            met = compute_metrics(y, m, top_thr, base_top_rate)
            if met:
                singles.append(
                    {
                        "rule_type": "categorical",
                        "conditions": [{"col": c, "op": "==", "val": val}],
                        **met,
                    }
                )

    # Numeric bins
    for c in (numeric_cols or []):
        x = pd.to_numeric(df_all[c], errors="coerce")
        try:
            bins = pd.qcut(x, q=num_bins, duplicates="drop")
        except Exception:
            continue
        vc = bins.value_counts(dropna=True)
        for bin_label, cnt in vc.items():
            if cnt < min_n_single:
                continue
            mask = (bins == bin_label)
            met = compute_metrics(y, mask, top_thr, base_top_rate)
            if not met:
                continue
            if hasattr(bin_label, "left"):
                left = float(bin_label.left)
                right = float(bin_label.right)
                closed = (
                    bins.cat.categories[0].closed if hasattr(bins, "cat") else "right"
                )
            else:
                left = right = np.nan
                closed = "right"
            singles.append(
                {
                    "rule_type": "numeric_bin",
                    "conditions": [
                        {
                            "col": c,
                            "op": "in_bin",
                            "val": {
                                "left": left,
                                "right": right,
                                "closed": str(closed),
                            },
                        }
                    ],
                    **met,
                }
            )

    df_singles = pd.DataFrame(singles)
    if not df_singles.empty:
        df_singles = df_singles.sort_values(
            ["Lift", "Ret_Mean"], ascending=[False, False]
        ).head(topk_single)
    else:
        df_singles = pd.DataFrame(
            columns=["rule_type", "conditions", "N", "Ret_Mean", "TopRate", "Lift"]
        )

    # Pair
    pairs: List[Dict] = []
    records = df_singles.to_dict(orient="records")
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            r1 = records[i]
            r2 = records[j]
            m1 = build_rule_mask(df_all, r1["conditions"])
            m2 = build_rule_mask(df_all, r2["conditions"])
            m = m1 & m2
            met = compute_metrics(y, m, top_thr, base_top_rate)
            if met and met["N"] >= min_n_pair:
                pairs.append(
                    {
                        "rule_type": "pair",
                        "conditions": r1["conditions"] + r2["conditions"],
                        **met,
                    }
                )

    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(
            ["Lift", "Ret_Mean"], ascending=[False, False]
        ).head(topk_pair)
    else:
        df_pairs = pd.DataFrame(
            columns=["rule_type", "conditions", "N", "Ret_Mean", "TopRate", "Lift"]
        )

    out = pd.concat(
        [df_singles.assign(level="single"), df_pairs.assign(level="pair")],
        ignore_index=True,
    )

    return out


# ============================================================
#  backtest4'ten: combo koşullarını parse / maske üret
# ============================================================

def detect_latest_backtest_csv(folder: str) -> Optional[str]:
    """
    Klasördeki en güncel GERCEK_BACKTEST5_TO_LATEST_YYYY-MM-DD.csv dosyasını bul.
    """
    pattern = re.compile(
        r"^GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.csv$",
        re.IGNORECASE,
    )
    candidates = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            date_str = m.group(1)
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue
            candidates.append((d, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, latest_file = candidates[-1]
    return os.path.join(folder, latest_file)


def detect_latest_multim4(folder: str) -> Optional[Tuple[str, str]]:
    """
    Klasördeki en güncel multim4_YYYY-MM-DD.csv dosyasını bulur.
    Returns: (date_str, full_path) tuple or None
    """
    pattern = re.compile(r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)
    candidates = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            date_str = m.group(1)
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                candidates.append((d, date_str, os.path.join(folder, fname)))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, date_str, file_path = candidates[-1]
    return (date_str, file_path)


def detect_latest_combo_files(folder: str, target_date: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Klasördeki COMBO_MINED5_SHORT_YYYY-MM-DD.csv ve COMBO_MINED5_MID_15GUN_YYYY-MM-DD.csv 
    dosyalarını bulur. Önce target_date'i arar, bulamazsa en güncel tarihi kullanır.
    
    Returns: (short_path, mid_path) tuple
    """
    pattern_short = re.compile(r"^COMBO_MINED5_SHORT_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)
    pattern_mid = re.compile(r"^COMBO_MINED5_MID_15GUN_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)
    
    short_candidates = []
    mid_candidates = []
    
    for fname in os.listdir(folder):
        m_short = pattern_short.match(fname)
        if m_short:
            date_str = m_short.group(1)
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                short_candidates.append((d, date_str, os.path.join(folder, fname)))
            except Exception:
                continue
        
        m_mid = pattern_mid.match(fname)
        if m_mid:
            date_str = m_mid.group(1)
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                mid_candidates.append((d, date_str, os.path.join(folder, fname)))
            except Exception:
                continue
    
    if not short_candidates or not mid_candidates:
        return None, None
    
    short_candidates.sort(key=lambda x: x[0])
    mid_candidates.sort(key=lambda x: x[0])
    
    # Try to find target_date first
    short_path = None
    mid_path = None
    
    if target_date:
        for d, date_str, path in short_candidates:
            if date_str == target_date:
                short_path = path
                break
        for d, date_str, path in mid_candidates:
            if date_str == target_date:
                mid_path = path
                break
    
    # Fallback to latest available, but both must have the same date
    if not short_path or not mid_path:
        # Find common dates between short and mid
        short_dates = {date_str: path for d, date_str, path in short_candidates}
        mid_dates = {date_str: path for d, date_str, path in mid_candidates}
        common_dates = sorted(set(short_dates.keys()) & set(mid_dates.keys()))
        
        if common_dates:
            latest_common_date = common_dates[-1]
            short_path = short_dates[latest_common_date]
            mid_path = mid_dates[latest_common_date]
            if target_date and latest_common_date != target_date:
                print(f"[Uyarı] Hedef tarih {target_date} için combo dosyaları bulunamadı. "
                      f"En güncel ortak tarih kullanılıyor: {latest_common_date}")
        else:
            return None, None
    
    return short_path, mid_path


def smart_to_numeric_b4(series: pd.Series) -> pd.Series:
    """backtest4'teki versiyon (isim çakışmasın diye)."""
    return smart_to_numeric(series)


def parse_conditions(cond_str: str) -> List[Dict[str, Any]]:
    """
    COMBO_MINED5 içindeki conditions string'ini Python listesine çevir.
    """
    if isinstance(cond_str, list):
        return cond_str
    if not isinstance(cond_str, str):
        return []
    cond_str = cond_str.strip()
    if not cond_str:
        return []
    try:
        parsed = ast.literal_eval(cond_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def build_rule_mask_from_conditions(df: pd.DataFrame, conditions: List[Dict[str, Any]]) -> pd.Series:
    """conditions listesini df üzerine uygula."""
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        if not isinstance(cond, dict):
            return pd.Series(False, index=df.index)
        col = cond.get("col")
        op = cond.get("op")
        val = cond.get("val")
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if op == "==":
            mask &= (df[col] == val)
        elif op == "in_bin":
            x = pd.to_numeric(df[col], errors="coerce")
            left = val.get("left", -np.inf)
            right = val.get("right", np.inf)
            closed = val.get("closed", "right")
            if closed == "right":
                mask &= (x > left) & (x <= right)
            else:
                mask &= (x >= left) & (x < right)
        else:
            mask &= False
    return mask


def _compute_pattern_score(
    y5: Optional[pd.Series],
    y15: Optional[pd.Series],
    symbols: Optional[pd.Series],
    complexity: int,
    horizon: str,
) -> float:
    if horizon == "5":
        if y5 is None or y5.empty:
            return -1e9
        HitThr = HIT_THR_5
        y = y5
    else:
        if y15 is None or y15.empty:
            return -1e9
        HitThr = HIT_THR_15
        y = y15

    N_total = len(y)
    if N_total == 0:
        return -1e9

    y = y.dropna()
    if y.empty:
        return -1e9

    N_pos = int((y >= HitThr).sum())
    N_neg = int((y < 0).sum())
    HitRate = N_pos / N_total
    AvgRet = float(y.mean())
    try:
        P20 = float(np.percentile(y, 20))
    except Exception:
        P20 = float(y.min())
    Pmin = float(y.min())
    NegRatio = N_neg / N_total

    if symbols is not None and not symbols.empty:
        syms = symbols.dropna().astype(str)
        NumSymbols = syms.nunique()
    else:
        NumSymbols = 1

    N_eff = np.log10(max(N_total, 1))
    Sym_eff = np.log10(max(NumSymbols, 1) + 1)

    RiskPen = max(0.0, -P20 / 5.0)
    NegPen = NegRatio * 100.0
    ComplexPen = max(0.0, complexity - 3)
    OverfitPen = max(0.0, (N_total / max(NumSymbols, 1)) - 10.0)

    score = (
        40.0 * HitRate
        + 30.0 * (AvgRet / max(HitThr, 1.0))
        + 10.0 * N_eff
        + 10.0 * Sym_eff
        - 10.0 * RiskPen
        - 5.0 * NegPen
        - 5.0 * ComplexPen
        - 10.0 * OverfitPen
    )
    return float(score)


def compute_combo_stats(df_all: pd.DataFrame, combos_df: pd.DataFrame) -> pd.DataFrame:
    """Her combo için 5g ve 15g getiriler üzerinden istatistikler hesapla (backtest4 + HitRate5)."""
    if combos_df is None or combos_df.empty:
        return pd.DataFrame()

    for c in [COL_MAX_5, COL_MAX_15]:
        if c in df_all.columns:
            df_all[c] = smart_to_numeric_b4(df_all[c])

    combos_df = combos_df.copy()
    combos_df["conditions_parsed"] = combos_df["conditions"].apply(parse_conditions)

    records = combos_df.to_dict(orient="records")
    out_rows: List[Dict[str, Any]] = []

    for rec in records:
        conditions = rec.get("conditions_parsed") or []
        if not conditions:
            continue

        mask = build_rule_mask_from_conditions(df_all, conditions)
        sub = df_all.loc[mask].copy()
        if sub.empty:
            continue

        row: Dict[str, Any] = {
            "rule_type": rec.get("rule_type", ""),
            "level": rec.get("level", ""),
            "N_raw": rec.get("N", None),
            "Lift_raw": rec.get("Lift", None),
            "Ret_Mean_raw": rec.get("Ret_Mean", None),
            "TopRate_raw": rec.get("TopRate", None),
            "N_rows": int(len(sub)),
            "conditions": conditions,  # Python listesi
        }

        # 5 gün
        y5_series = None
        if COL_MAX_5 in sub.columns:
            y5 = pd.to_numeric(sub[COL_MAX_5], errors="coerce").dropna()
            if not y5.empty:
                row["N_5"] = int(len(y5))
                row["Mean_5"] = float(y5.mean())
                row["Std_5"] = float(y5.std(ddof=0))
                row["Min_5"] = float(y5.min())
                row["NegRatio_5"] = float((y5 < 0).mean())
                y5_series = y5
            else:
                row["N_5"] = 0
        else:
            row["N_5"] = 0

        # 5g HitRate ve NumSymbols_5
        if y5_series is not None and len(y5_series) > 0:
            N5 = len(y5_series)
            N_pos_20 = int((y5_series >= HIT_THR_5).sum())
            HitRate5 = N_pos_20 / N5
            row["HitRate5"] = HitRate5
            if COL_SYMBOL in sub.columns:
                row["NumSymbols_5"] = int(sub[COL_SYMBOL].dropna().astype(str).nunique())
            else:
                row["NumSymbols_5"] = 1
            row["_y5_series"] = y5_series
        else:
            row["HitRate5"] = 0.0
            row["NumSymbols_5"] = 0

        # 15 gün
        if COL_MAX_15 in sub.columns:
            y15 = pd.to_numeric(sub[COL_MAX_15], errors="coerce").dropna()
            if not y15.empty:
                row["N_15"] = int(len(y15))
                row["Mean_15"] = float(y15.mean())
                row["Std_15"] = float(y15.std(ddof=0))
                row["Min_15"] = float(y15.min())
                row["NegRatio_15"] = float((y15 < 0).mean())
                row["_y15_series"] = y15
            else:
                row["N_15"] = 0
        else:
            row["N_15"] = 0

        complexity = len(conditions)
        row["Complexity"] = complexity
        syms = sub[COL_SYMBOL] if COL_SYMBOL in sub.columns else None

        if row.get("N_5", 0) > 0:
            row["PatternScore5"] = _compute_pattern_score(
                y5=row.get("_y5_series"),
                y15=None,
                symbols=syms,
                complexity=complexity,
                horizon="5",
            )
        else:
            row["PatternScore5"] = -1e9

        if row.get("N_15", 0) > 0:
            row["PatternScore15"] = _compute_pattern_score(
                y5=None,
                y15=row.get("_y15_series"),
                symbols=syms,
                complexity=complexity,
                horizon="15",
            )
        else:
            row["PatternScore15"] = -1e9

        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    df_stats = pd.DataFrame(out_rows)

    if any(c in df_stats.columns for c in ["_y5_series", "_y15_series"]):
        df_stats = df_stats.drop(columns=[c for c in ["_y5_series", "_y15_series"] if c in df_stats.columns])

    if "N_5" in df_stats.columns:
        risk_5 = df_stats["Std_5"].fillna(0.0)
        mean_5 = df_stats["Mean_5"].fillna(0.0)
        df_stats["Skor_5D_Dusuk"] = mean_5 - ALPHA_5D_LOW * risk_5
        df_stats["Skor_5D_Yuksek"] = mean_5 - GAMMA_5D_HIGH * risk_5

    if "N_15" in df_stats.columns:
        risk_15 = df_stats["Std_15"].fillna(0.0)
        mean_15 = df_stats["Mean_15"].fillna(0.0)
        df_stats["Skor_15D_Dusuk"] = mean_15 - BETA_15D_LOW * risk_15

    return df_stats


def select_best_combos(df_stats: pd.DataFrame):
    """
    İstatistiklerden 3 hedef için en iyi combo setlerini seç (backtest4 + PatternScore).
    NOT: 5g düşük risk için HitRate5 filtresi kullanılmıyor, sadece sıralamada kullanılıyor.
    """
    if df_stats.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 5 gün düşük risk
    cond_5_low = (
        (df_stats["N_5"] >= max(MIN_N_5D_LOW, MIN_N5_HITFILTER))   # en az 50 örnek
        & (df_stats["Mean_5"] >= THRESH_5D_MIN)                    # ortalama getiri >= %10
        & (df_stats["NegRatio_5"] <= MAX_NEG_RATIO_5D_LOW)         # negatif oranı <= 0.45
        # HitRate filtresi veri yetersizliği sebebiyle devre dışı:
        # & (df_stats["HitRate5"] >= MIN_HITRATE5)
        & (df_stats["NumSymbols_5"] >= MIN_SYMBOLS5)               # en az 5 farklı sembol
    )

    if "PatternScore5" in df_stats.columns:
        sort_cols_5_low = ["HitRate5", "Mean_5", "PatternScore5"]
    else:
        sort_cols_5_low = ["HitRate5", "Mean_5", "Skor_5D_Dusuk"]

    best_5_low = (
        df_stats[cond_5_low]
        .sort_values(sort_cols_5_low, ascending=[False, False, False])
        .head(20)
        .reset_index(drop=True)
    )

    # 5 gün yüksek risk
    cond_5_any = df_stats["N_5"] >= MIN_N_5D_HIGH
    cand_5 = df_stats[cond_5_any].copy()
    if not cand_5.empty:
        thr_mean5 = cand_5["Mean_5"].quantile(0.7)
        cond_5_high = cond_5_any & (df_stats["Mean_5"] >= thr_mean5)
        if "PatternScore5" in df_stats.columns:
            sort_cols_5_high = ["Mean_5", "PatternScore5"]
        else:
            sort_cols_5_high = ["Skor_5D_Yuksek", "Mean_5"]

        best_5_high = (
            df_stats[cond_5_high]
            .sort_values(sort_cols_5_high, ascending=[False, False])
            .head(20)
            .reset_index(drop=True)
        )
    else:
        best_5_high = pd.DataFrame()

    # 15 gün düşük risk (EŞİK ARTIK %20)
    cond_15_low = (
        (df_stats["N_15"] >= MIN_N_15D_LOW)
        & (df_stats["Mean_15"] >= THRESH_15D_MIN)
        & (df_stats["NegRatio_15"] <= MAX_NEG_RATIO_15D_LOW)
    )
    if "PatternScore15" in df_stats.columns:
        sort_cols_15_low = ["PatternScore15", "Mean_15"]
    else:
        sort_cols_15_low = ["Skor_15D_Dusuk", "Mean_15"]

    best_15_low = (
        df_stats[cond_15_low]
        .sort_values(sort_cols_15_low, ascending=[False, False])
        .head(20)
        .reset_index(drop=True)
    )

    return best_5_low, best_5_high, best_15_low


def apply_best_combos_to_backtest(
    df_all: pd.DataFrame,
    df_best_5_low: pd.DataFrame,
    df_best_5_high: pd.DataFrame,
    df_best_15_low: pd.DataFrame,
    today_str: str,
    folder: str,
):
    """Seçilen combo setlerini GERCEK_BACKTEST5 satırlarına uygula ve listeleri üret (backtest4 mantığı)."""

    def _apply_combo_set(df_src: pd.DataFrame, df_combos: pd.DataFrame, horizon: str):
        df_src = df_src.copy()

        if horizon in ("5_low", "5_high"):
            col_mean = "Beklenen_5g_Getiri_%"
            col_risk = "Tahmini_Risk_5g"
            col_id = "Kural_ID_5g"
        else:
            col_mean = "Beklenen_15g_Getiri_%"
            col_risk = "Tahmini_Risk_15g"
            col_id = "Kural_ID_15g"

        df_src[col_mean] = None
        df_src[col_risk] = None
        df_src[col_id] = None

        if df_combos is None or df_combos.empty:
            return df_src

        records = df_combos.to_dict(orient="records")

        best_means = []
        best_risks = []
        best_ids = []
        
        # Diagnostic: Track missing columns and failed conditions
        missing_columns = set()
        total_rows = 0
        total_combos = len(records)
        rows_with_matches = 0
        condition_failures = {"missing_col": 0, "value_mismatch": 0, "bin_mismatch": 0}

        for _, row in df_src.iterrows():
            total_rows += 1
            best_mean = None
            best_risk = None
            best_id = None
            row_matched_any = False

            for idx, combo in enumerate(records):
                conditions = combo.get("conditions", [])
                if not isinstance(conditions, list):
                    continue

                ok = True
                fail_reason = None
                for cond in conditions:
                    if not isinstance(cond, dict):
                        ok = False
                        break
                    col = cond.get("col")
                    op = cond.get("op")
                    val = cond.get("val")
                    if col not in df_src.columns:
                        missing_columns.add(col)
                        condition_failures["missing_col"] += 1
                        ok = False
                        fail_reason = "missing_col"
                        break
                    v = row.get(col, np.nan)
                    if op == "==":
                        if pd.isna(v) or v != val:
                            condition_failures["value_mismatch"] += 1
                            ok = False
                            fail_reason = "value_mismatch"
                            break
                    elif op == "in_bin":
                        vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
                        if pd.isna(vv):
                            ok = False
                            fail_reason = "value_mismatch"
                            break
                        left = val.get("left", -np.inf)
                        right = val.get("right", np.inf)
                        closed = val.get("closed", "right")
                        if closed == "right":
                            if not (vv > left and vv <= right):
                                condition_failures["bin_mismatch"] += 1
                                ok = False
                                fail_reason = "bin_mismatch"
                                break
                        else:
                            if not (vv >= left and vv < right):
                                condition_failures["bin_mismatch"] += 1
                                ok = False
                                fail_reason = "bin_mismatch"
                                break
                    else:
                        ok = False
                        break

                if not ok:
                    continue

                if horizon in ("5_low", "5_high"):
                    m = combo.get("Mean_5", None)
                    r = combo.get("Std_5", None)
                else:
                    m = combo.get("Mean_15", None)
                    r = combo.get("Std_15", None)

                if m is None:
                    continue

                if (best_mean is None) or (m > best_mean):
                    best_mean = m
                    best_risk = r
                    best_id = f"combo_{idx}"
                    row_matched_any = True

            if row_matched_any:
                rows_with_matches += 1
                
            best_means.append(best_mean)
            best_risks.append(best_risk)
            best_ids.append(best_id)

        df_src[col_mean] = best_means
        df_src[col_risk] = best_risks
        df_src[col_id] = best_ids
        
        # Print diagnostic info
        print(f"[Debug] {horizon}: {total_rows} satır, {total_combos} combo kontrol edildi")
        print(f"        {rows_with_matches} satırda en az 1 combo eşleşti")
        if missing_columns:
            print(f"[Uyarı] {len(missing_columns)} kolon eksik:")
            print(f"        {', '.join(sorted(list(missing_columns))[:15])}")
        if condition_failures["missing_col"] > 0 or condition_failures["value_mismatch"] > 0:
            print(f"[Debug] Başarısızlık nedenleri:")
            print(f"        Eksik kolon: {condition_failures['missing_col']} kez")
            print(f"        Değer uyuşmazlığı: {condition_failures['value_mismatch']} kez")
            print(f"        Bin aralığı dışı: {condition_failures['bin_mismatch']} kez")

        return df_src

    # 5g düşük risk
    df_5_low = _apply_combo_set(df_all, df_best_5_low, horizon="5_low")
    if "Beklenen_5g_Getiri_%" in df_5_low.columns:
        list1 = df_5_low.dropna(subset=["Beklenen_5g_Getiri_%"]).copy()
        list1 = list1[list1["Beklenen_5g_Getiri_%"] >= THRESH_5D_MIN]

        # Teknik kolonlar sayıya çevrilip sıralamaya eklenir
        for col in ["MACD_Signal_5g", "PUANLAMA_V4_5g"]:
            if col in list1.columns:
                list1[col] = pd.to_numeric(list1[col], errors="coerce")

        sort_cols = ["Tahmini_Risk_5g", "Beklenen_5g_Getiri_%"]
        ascending = [True, False]

        if "PUANLAMA_V4_5g" in list1.columns:
            sort_cols.append("PUANLAMA_V4_5g")
            ascending.append(False)
        if "MACD_Signal_5g" in list1.columns:
            sort_cols.append("MACD_Signal_5g")
            ascending.append(False)

        list1 = list1.sort_values(sort_cols, ascending=ascending)
    else:
        list1 = pd.DataFrame()

    # 5g yüksek risk – 20 satır (RSI_5g filtresi + teknik sıralama)
    df_5_high = _apply_combo_set(df_all, df_best_5_high, horizon="5_high")
    if "Beklenen_5g_Getiri_%" in df_5_high.columns:
        list2 = df_5_high.dropna(subset=["Beklenen_5g_Getiri_%"]).copy()

        # Teknik kolonları sayıya çevir
        for col in ["RSI_5g", "MACD_Signal_5g", "PUANLAMA_V4_5g"]:
            if col in list2.columns:
                list2[col] = pd.to_numeric(list2[col], errors="coerce")

        # RSI filtresi: 40–80
        if "RSI_5g" in list2.columns:
            before = len(list2)
            list2 = list2[(list2["RSI_5g"] >= 40) & (list2["RSI_5g"] <= 80)].copy()
            after = len(list2)
            print(f"[Bilgi] backtest5: 5g yüksek risk RSI filtresi (40<=RSI_5g<=80): {before} -> {after} satır.")

        # Sıralama: Beklenen_5g_Getiri_% ↓, MACD_Signal_5g ↓, PUANLAMA_V4_5g ↓
        sort_cols = ["Beklenen_5g_Getiri_%"]
        ascending = [False]
        if "MACD_Signal_5g" in list2.columns:
            sort_cols.append("MACD_Signal_5g")
            ascending.append(False)
        if "PUANLAMA_V4_5g" in list2.columns:
            sort_cols.append("PUANLAMA_V4_5g")
            ascending.append(False)

        list2 = list2.sort_values(sort_cols, ascending=ascending).head(20)
    else:
        list2 = pd.DataFrame()

    # 15g düşük risk – EŞİK ARTIK %20
    df_15_low = _apply_combo_set(df_all, df_best_15_low, horizon="15")
    if "Beklenen_15g_Getiri_%" in df_15_low.columns:
        list3 = df_15_low.dropna(subset=["Beklenen_15g_Getiri_%"]).copy()
        list3 = list3[list3["Beklenen_15g_Getiri_%"] >= THRESH_15D_MIN]
        list3 = list3.sort_values(
            ["Tahmini_Risk_15g", "Beklenen_15g_Getiri_%"],
            ascending=[True, False],
        )
    else:
        list3 = pd.DataFrame()

    # Ortak kolonlar
    cols_common = []
    if COL_SYMBOL in df_all.columns:
        cols_common.append(COL_SYMBOL)
    if COL_ANALIZ_TARIHI in df_all.columns:
        cols_common.append(COL_ANALIZ_TARIHI)
    if COL_FIYAT_SON in df_all.columns:
        cols_common.append(COL_FIYAT_SON)
    if COL_FIYAT_ANALIZ in df_all.columns and COL_FIYAT_ANALIZ not in cols_common:
        cols_common.append(COL_FIYAT_ANALIZ)

    def _save_list(df_list: pd.DataFrame, cols: List[str], filename: str):
        if df_list is None or df_list.empty:
            print(f"[Bilgi] {filename} için uygun satır bulunamadı.")
            return
        used_cols = [c for c in cols if c in df_list.columns]
        out = df_list[used_cols].reset_index(drop=True)
        path = os.path.join(folder, filename)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[ÇIKTI] {path}")

    # 3 ana liste
    _save_list(
        list1,
        cols_common + ["Beklenen_5g_Getiri_%", "Tahmini_Risk_5g", "Kural_ID_5g"],
        LIST_1_TEMPLATE.format(date=today_str),
    )
    _save_list(
        list2,
        cols_common + ["Beklenen_5g_Getiri_%", "Tahmini_Risk_5g", "Kural_ID_5g"],
        LIST_2_TEMPLATE.format(date=today_str),
    )
    _save_list(
        list3,
        cols_common + ["Beklenen_15g_Getiri_%", "Tahmini_Risk_15g", "Kural_ID_15g"],
        LIST_3_TEMPLATE.format(date=today_str),
    )

    # Riskli birleşik liste (Liste2 + Liste3)
    if not list2.empty or not list3.empty:
        merge_keys = []
        if COL_SYMBOL in df_all.columns:
            merge_keys.append(COL_SYMBOL)
        if COL_ANALIZ_TARIHI in df_all.columns:
            merge_keys.append(COL_ANALIZ_TARIHI)

        if merge_keys:
            l2 = list2.copy()
            l3 = list3.copy()
            l2 = l2.rename(
                columns={
                    "Beklenen_5g_Getiri_%": "Beklenen_5g",
                    "Tahmini_Risk_5g": "Risk_5g",
                    "Kural_ID_5g": "Kural_5g",
                }
            )
            l3 = l3.rename(
                columns={
                    "Beklenen_15g_Getiri_%": "Beklenen_15g",
                    "Tahmini_Risk_15g": "Risk_15g",
                    "Kural_ID_15g": "Kural_15g",
                }
            )

            on_cols = merge_keys.copy()
            if COL_FIYAT_SON in df_all.columns and COL_FIYAT_SON in l2.columns and COL_FIYAT_SON in l3.columns:
                on_cols.append(COL_FIYAT_SON)

            combined = pd.merge(
                l2,
                l3,
                how="outer",
                on=on_cols,
                suffixes=("_5g", "_15g"),
            )
        else:
            combined = pd.concat(
                [
                    list2.rename(
                        columns={
                            "Beklenen_5g_Getiri_%": "Beklenen_5g",
                            "Tahmini_Risk_5g": "Risk_5g",
                            "Kural_ID_5g": "Kural_5g",
                        }
                    ).assign(ListType="5G_RISKLI"),
                    list3.rename(
                        columns={
                            "Beklenen_15g_Getiri_%": "Beklenen_15g",
                            "Tahmini_Risk_15g": "Risk_15g",
                            "Kural_ID_15g": "Kural_15g",
                        }
                    ).assign(ListType="15G_DUSUK_RISK"),
                ],
                ignore_index=True,
            )

        for col in ["Beklenen_5g", "Risk_5g", "Beklenen_15g", "Risk_15g"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")

        def _row_score(row):
            b5 = row.get("Beklenen_5g", np.nan)
            b15 = row.get("Beklenen_15g", np.nan)
            if not pd.isna(b15):
                return b15
            return b5

        combined["Secim_Skoru"] = combined.apply(_row_score, axis=1)

        if COL_SYMBOL in combined.columns:
            combined = (
                combined.sort_values("Secim_Skoru", ascending=False)
                .drop_duplicates(subset=[COL_SYMBOL])
            )

        combined = combined.sort_values("Secim_Skoru", ascending=False).reset_index(drop=True)

        combined_path = os.path.join(
            folder, COMBINED_RISKY_CSV_TEMPLATE.format(date=today_str)
        )
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"[ÇIKTI] {combined_path}")

        try:
            import openpyxl  # noqa: F401

            xlsx_path = os.path.join(
                folder, COMBINED_RISKY_XLSX_TEMPLATE.format(date=today_str)
            )
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
                combined.to_excel(w, index=False, sheet_name="RISKLI_5G_15G")
            print(f"[XLSX] {xlsx_path}")
        except ModuleNotFoundError:
            print("[Uyarı] openpyxl yok, birleşik XLSX yazılamadı.")

    else:
        print("[Bilgi] Riskli 5g veya 15g listeleri boş, birleşik dosya üretilmedi.")


# ============================================================
#  GERÇEK BACKTEST5: multim4 -> forward getiriler (backtest3)
# ============================================================

def run_backtest_v5(cfg: Dict):
    folder = cfg["dir"]
    if not os.path.isdir(folder):
        print(f"[HATA] Klasör yok: {folder}")
        return

    files, files_with_dates = select_multim4_base_files(
        folder, cfg["start_date"], cfg["end_date"], cfg["last_n_files"]
    )
    print(f"Alınan multim4 taban dosyaları ({len(files)}): {files}")

    mid_horizon = int(cfg.get("mid_horizon_days", 15))
    if len(files) < mid_horizon + 1:
        print(f"Yeterli sayıda (en az {mid_horizon + 1}) multim4 dosyası bulunamadı.")
        return

    # En son dosya (multim4) ve kolonları
    latest_file_info = None
    for f, d in reversed(files_with_dates):
        path = os.path.join(folder, f)
        df_cand = _read_csv_any(path)
        if df_cand is not None and not df_cand.empty:
            sc, pc = resolve_symbol_and_price_columns(df_cand)
            if sc and pc:
                latest_file_info = {
                    "file": f,
                    "df": df_cand,
                    "sym_col": sc,
                    "price_col": pc,
                }
                break

    if latest_file_info is None:
        print("[HATA] Hiçbir multim4 dosyasında Sembol ve Fiyat kolonları bulunamadı.")
        return

    latest_df = latest_file_info["df"]
    latest_sym_col = latest_file_info["sym_col"]
    latest_price_col = latest_file_info["price_col"]
    print(
        f"[Bilgi] Latest multim4 dosya: {latest_file_info['file']} | "
        f"Sembol: {latest_sym_col} | Fiyat: {latest_price_col}"
    )

    # Geçmiş fiyat verilerini önbelleğe al
    print("[Bilgi] Geçmiş veriler önbelleğe alınıyor...")
    price_cache: Dict[str, Dict[str, float]] = {}  # {dosya_ismi: {sembol: kapanış_fiyatı}}

    for f, d in files_with_dates:
        path = os.path.join(folder, f)
        df = _read_csv_any(path)
        if df is None:
            continue
        sc, pc = resolve_symbol_and_price_columns(df)
        if not sc or not pc:
            continue
        df[sc] = df[sc].astype(str).str.strip().str.upper()
        df[pc] = smart_to_numeric(df[pc])
        price_cache[f] = dict(zip(df[sc], df[pc]))

    all_rows: List[Dict] = []
    backtest_files = files_with_dates[:-mid_horizon]

    print(f"[Bilgi] {len(backtest_files)} gün üzerinde backtest yapılacak...")
    for i, (f_base, d_base) in enumerate(backtest_files):
        df_base = _read_csv_any(os.path.join(folder, f_base))
        if df_base is None:
            continue
        sc_base, pc_base = resolve_symbol_and_price_columns(df_base)
        if not sc_base or not pc_base:
            continue

        df_base[sc_base] = df_base[sc_base].astype(str).str.strip().str.upper()
        df_base[pc_base] = smart_to_numeric(df_base[pc_base])

        future_files = [files_with_dates[i + k][0] for k in range(1, mid_horizon + 1)]

        for _, row in df_base.iterrows():
            sym = row.get(sc_base, "")
            alis = row.get(pc_base)
            if not sym or pd.isna(alis):
                continue

            future_price_list: List[Tuple[int, float]] = []
            for day_idx, f_future in enumerate(future_files, start=1):
                price = price_cache.get(f_future, {}).get(sym)
                if price is not None and pd.notna(price):
                    future_price_list.append((day_idx, float(price)))

            if not future_price_list:
                continue

            prices_all = [p for _, p in future_price_list]
            if len(prices_all) < 5:
                continue

            prices_5 = prices_all[:5]
            fiyat_zirve_5 = max(prices_5)
            fiyat_dip_5 = min(prices_5)
            fiyat_1hafta_sonu = prices_5[-1]

            fiyat_zirve_15 = max(prices_all)
            fiyat_dip_15 = min(prices_all)
            fiyat_15gun_sonu = prices_all[-1]

            five_day_list = future_price_list[:5]
            gun_en_yuksek_5 = max(five_day_list, key=lambda x: x[1])[0]
            gun_en_dusuk_5 = min(five_day_list, key=lambda x: x[1])[0]

            rec = row.to_dict()
            rec[COL_ANALIZ_TARIHI] = d_base.strftime("%Y-%m-%d") if d_base else ""
            rec[COL_SYMBOL] = sym
            rec[COL_FIYAT_ANALIZ] = float(alis)

            # 5 günlük metrikler
            if alis > 0:
                max_getiri_5 = (fiyat_zirve_5 - alis) / alis * 100.0
                getiri_1hafta = (fiyat_1hafta_sonu - alis) / alis * 100.0
            else:
                max_getiri_5 = 0.0
                getiri_1hafta = 0.0

            if fiyat_zirve_5 > 0:
                zirve_kayip_5 = (fiyat_zirve_5 - fiyat_1hafta_sonu) / fiyat_zirve_5 * 100.0
            else:
                zirve_kayip_5 = 0.0

            rec[COL_MAX_5] = max_getiri_5
            rec["Getiri_1Hafta_Sonu_%"] = getiri_1hafta
            rec["Zirve_Kayip_%"] = zirve_kayip_5

            rec["Fiyat_EnYuksek_5Gun"] = float(fiyat_zirve_5)
            rec["Fiyat_EnDusuk_5Gun"] = float(fiyat_dip_5)
            rec["Max_Getiri_5Gun_%"] = max_getiri_5
            rec["Gun_EnYuksek"] = gun_en_yuksek_5
            rec["Gun_EnDusuk"] = gun_en_dusuk_5

            # 15 günlük metrikler
            if alis > 0:
                max_getiri_15 = (fiyat_zirve_15 - alis) / alis * 100.0
                getiri_15gun_sonu = (fiyat_15gun_sonu - alis) / alis * 100.0
            else:
                max_getiri_15 = 0.0
                getiri_15gun_sonu = 0.0

            if fiyat_zirve_15 > 0:
                zirve_kayip_15 = (fiyat_zirve_15 - fiyat_15gun_sonu) / fiyat_zirve_15 * 100.0
            else:
                zirve_kayip_15 = 0.0

            rec[COL_MAX_15] = max_getiri_15
            rec["Getiri_15Gun_Sonu_%"] = getiri_15gun_sonu
            rec["Zirve_Kayip_15Gun_%"] = zirve_kayip_15
            rec["Fiyat_EnYuksek_15Gun"] = float(fiyat_zirve_15)
            rec["Fiyat_EnDusuk_15Gun"] = float(fiyat_dip_15)

            all_rows.append(rec)

    if not all_rows:
        print("Hiç backtest satırı üretilemedi. Dosya yapılarını kontrol edin.")
        return

    df_all = pd.DataFrame(all_rows)

    # Hedef kolonlar ve Top_Grup etiketleri (backtest3 ile aynı)
    y_col_short = COL_MAX_5
    if y_col_short not in df_all.columns or df_all[y_col_short].notna().sum() == 0:
        print(f"Geçerli '{y_col_short}' verisi yok. Kural madenciliği atlanıyor.")
        return

    df_all[y_col_short] = pd.to_numeric(df_all[y_col_short], errors="coerce")
    q_short = 1.0 - cfg["top_quantile"]
    thr_short = df_all[y_col_short].quantile(q_short)
    df_all["Top_Grup"] = (df_all[y_col_short] >= thr_short).astype(int)

    print(
        f"[ÖZET KISA] Satır: {len(df_all)} | "
        f"Ort.Max_Getiri%={df_all[y_col_short].mean():.2f} | "
        f"Medyan={df_all[y_col_short].median():.2f} | "
        f"Top_eşik={thr_short:.2f}"
    )

    y_col_mid = COL_MAX_15
    has_mid = y_col_mid in df_all.columns and df_all[y_col_mid].notna().sum() > 0
    if has_mid:
        df_all[y_col_mid] = pd.to_numeric(df_all[y_col_mid], errors="coerce")
        q_mid = 1.0 - cfg.get("top_quantile_mid", cfg["top_quantile"])
        thr_mid = df_all[y_col_mid].quantile(q_mid)
        df_all["Top_Grup_15Gun"] = (df_all[y_col_mid] >= thr_mid).astype(int)
        print(
            f"[ÖZET ORTA] Ort.Max_Getiri_15Gun%={df_all[y_col_mid].mean():.2f} | "
            f"Medyan={df_all[y_col_mid].median():.2f} | "
            f"Top_eşik_15Gun={thr_mid:.2f}"
        )
    else:
        print("[Uyarı] Orta vade (15 gün) metrikleri için geçerli veri yok.")

    # --- GERÇEK BACKTEST5 CSV ---
    today = datetime.now().strftime("%Y-%m-%d")
    out_rows = os.path.join(folder, f"GERCEK_BACKTEST5_TO_LATEST_{today}.csv")
    df_all.to_csv(out_rows, index=False, encoding="utf-8-sig")
    print(f"[ÇIKTI] {out_rows}")

    # --- v5 mining: performans kolonları feature olarak kullanılmıyor ---
    combos_df_short = pd.DataFrame()
    combos_df_mid = pd.DataFrame()

    if cfg["mine_combos"]:
        ignore = {
            # ID / tarih / sembol
            COL_ANALIZ_TARIHI,
            COL_SYMBOL,
            COL_FIYAT_ANALIZ,
            # kısa vade performans & ekleri
            "Max_Getiri_%",
            "Getiri_1Hafta_Sonu_%",
            "Zirve_Kayip_%",
            "Top_Grup",
            "Fiyat_EnYuksek_5Gun",
            "Fiyat_EnDusuk_5Gun",
            "Max_Getiri_5Gun_%",
            "Gun_EnYuksek",
            "Gun_EnDusuk",
            # orta vade performans & ekleri
            "Max_Getiri_15Gun_%",
            "Getiri_15Gun_Sonu_%",
            "Zirve_Kayip_15Gun_%",
            "Fiyat_EnYuksek_15Gun",
            "Fiyat_EnDusuk_15Gun",
            "Top_Grup_15Gun",
        }

        numeric_cols, binary_cols, categorical_cols = classify_columns(
            df_all, ignore, cfg["max_cat_unique"]
        )

        print(f"[Bilgi] Mining feature sayıları: numeric={len(numeric_cols)}, binary={len(binary_cols)}, cat={len(categorical_cols)}")

        # --- Kısa vade combos: COMBO_MINED5_SHORT_YYYY-MM-DD.csv ---
        print(f"[Bilgi] '{y_col_short}' (kısa vade) hedefine göre kurallar çıkarılıyor (v5, performans kolonları hariç)...")
        combos_df_short = mine_combos(
            df_all=df_all,
            y_col=y_col_short,
            top_quantile=cfg["top_quantile"],
            binary_cols=binary_cols,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            num_bins=cfg["num_bins"],
            min_n_single=cfg["combo_min_n_single"],
            min_n_pair=cfg["combo_min_n_pair"],
            topk_single=cfg["combo_topk_single"],
            topk_pair=cfg["combo_topk_pair"],
        )
        out_combo_short = os.path.join(folder, f"COMBO_MINED5_SHORT_{today}.csv")
        combos_export_short = combos_df_short.copy()
        combos_export_short["conditions"] = combos_export_short["conditions"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        combos_export_short.to_csv(out_combo_short, index=False, encoding="utf-8-sig")
        print(f"[ÇIKTI] {out_combo_short}")

        # --- Orta vade combos: COMBO_MINED5_MID_15GUN_YYYY-MM-DD.csv ---
        if has_mid:
            print(f"[Bilgi] '{y_col_mid}' (orta vade 15 gün) hedefine göre kurallar çıkarılıyor (v5, performans kolonları hariç)...")
            combos_df_mid = mine_combos(
                df_all=df_all,
                y_col=y_col_mid,
                top_quantile=cfg.get("top_quantile_mid", cfg["top_quantile"]),
                binary_cols=binary_cols,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                num_bins=cfg["num_bins"],
                min_n_single=cfg["combo_min_n_single"],
                min_n_pair=cfg["combo_min_n_pair"],
                topk_single=cfg["combo_topk_single"],
                topk_pair=cfg["combo_topk_pair"],
            )
            out_combo_mid = os.path.join(folder, f"COMBO_MINED5_MID_15GUN_{today}.csv")
            combos_export_mid = combos_df_mid.copy()
            combos_export_mid["conditions"] = combos_export_mid["conditions"].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
            )
            combos_export_mid.to_csv(out_combo_mid, index=False, encoding="utf-8-sig")
            print(f"[ÇIKTI] {out_combo_mid}")
        else:
            print("[Bilgi] Orta vade yok, COMBO_MINED5_MID_15GUN yazılmadı.")

    # XLSX backtest özet (opsiyonel)
    if cfg["export_xlsx"]:
        try:
            import openpyxl  # noqa: F401

            xlsx = os.path.join(folder, f"GERCEK_BACKTEST5_TO_LATEST_{today}.xlsx")
            with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
                df_all.to_excel(w, index=False, sheet_name="Satirlar")
                if not combos_df_short.empty:
                    combos_short_export = combos_df_short.copy()
                    combos_short_export["conditions"] = combos_short_export[
                        "conditions"
                    ].astype(str)
                    combos_short_export.to_excel(w, index=False, sheet_name="Combos5Short")
                if not combos_df_mid.empty:
                    combos_mid_export = combos_df_mid.copy()
                    combos_mid_export["conditions"] = combos_mid_export[
                        "conditions"
                    ].astype(str)
                    combos_mid_export.to_excel(w, index=False, sheet_name="Combos5Mid15Gun")
            print(f"[XLSX] {xlsx}")
        except ModuleNotFoundError:
            print("[Uyarı] openpyxl yok, backtest5 XLSX yazılmadı.")

    print("[Tamamlandı] v5: 5g + 15g backtest + (combo mining v5) tamamlandı.")


def run_selection_v5_from_backtest():
    """
    backtest4 mantığını, v5 backtest & v5 combo dosyalarıyla çalıştır.
    Yani:
      - En güncel multim4_YYYY-MM-DD.csv (BUGÜNÜN VERİSİ)
      - COMBO_MINED5_SHORT_YYYY-MM-DD.csv (GEÇMİŞTEN ÖĞREN İLEN PATTERN'LER)
      - COMBO_MINED5_MID_15GUN_YYYY-MM-DD.csv
    kullanarak 3 liste + birleşik riskli liste üret.
    
    ÖNEMLİ: Öğrenilen pattern'leri BUGÜNÜN multim4 verisine uygular.
    """
    # 1. En güncel multim4 dosyasını yükle (BUGÜNÜN VERİSİ)
    multim_info = detect_latest_multim4(BASE_DIR)
    if not multim_info:
        print("[Hata] multim4_YYYY-MM-DD.csv dosyası bulunamadı. Seçim yapılamaz.")
        return
    
    selection_date_str, multim_path = multim_info
    print(f"[Bilgi] Seçim tarihi (en güncel multim4): {selection_date_str}")
    print(f"[Bilgi] Multim4 dosyası: {multim_path}")

    # 2. Bugünün multim4 verisini yükle
    df_today = _read_csv_any(multim_path)
    if df_today is None or df_today.empty:
        print(f"[Hata] Multim4 dosyası okunamadı veya boş: {multim_path}")
        return
    
    # Symbol ve price kolonlarını bul
    sym_col, price_col = resolve_symbol_and_price_columns(df_today)
    if not sym_col or not price_col:
        print(f"[Hata] Multim4 dosyasında Sembol ve Fiyat kolonları bulunamadı.")
        return
    
    # Sembol normalizasyonu
    df_today[sym_col] = df_today[sym_col].astype(str).str.strip().str.upper()
    
    # Analiz tarihi ekle (bu satırlar bugünün analizi için)
    df_today[COL_ANALIZ_TARIHI] = selection_date_str
    
    # Kolon isimlendirmelerini standartlaştır
    if sym_col != COL_SYMBOL:
        df_today = df_today.rename(columns={sym_col: COL_SYMBOL})
    if price_col != COL_FIYAT_SON:
        df_today = df_today.rename(columns={price_col: COL_FIYAT_SON})
    
    print(f"[Bilgi] Bugünün multim4 verisi yüklendi: {len(df_today)} satır")
    
    # Use today's data for selection
    df_selection = df_today

    
    # 3. Combo dosyalarını bul - en iyi combolar zaten seçilmiş olmalı
    # Önce BEST_COMBOS dosyalarını dene
    best_5_low_path = os.path.join(BASE_DIR, BEST_5D_LOW_TEMPLATE.format(date=selection_date_str))
    best_5_high_path = os.path.join(BASE_DIR, BEST_5D_HIGH_TEMPLATE.format(date=selection_date_str))
    best_15_low_path = os.path.join(BASE_DIR, BEST_15D_LOW_TEMPLATE.format(date=selection_date_str))
    
    # Eğer BEST_COMBOS dosyaları varsa onları kullan
    if os.path.isfile(best_5_low_path) and os.path.isfile(best_5_high_path):
        print(f"[Bilgi] Hazır BEST_COMBOS dosyaları kullanılıyor...")
        best_5_low = pd.read_csv(best_5_low_path)
        best_5_high = pd.read_csv(best_5_high_path)
        if os.path.isfile(best_15_low_path):
            best_15_low = pd.read_csv(best_15_low_path)
        else:
            best_15_low = pd.DataFrame()
    else:
        # BEST_COMBOS yoksa, COMBO_MINED5 dosyalarından yükle ve backtest üzerinden değerlendir
        print(f"[Bilgi] BEST_COMBOS dosyaları bulunamadı, combo dosyaları değerlendiriliyor...")
        
        combo_short_path, combo_mid_path = detect_latest_combo_files(BASE_DIR, selection_date_str)
        
        if not combo_short_path or not combo_mid_path:
            print(f"[Hata] COMBO_MINED5 dosyaları bulunamadı (hedef tarih: {selection_date_str}).")
            print("       Lütfen backtest5.py'yi --run-backtest ile çalıştırarak combo dosyalarını oluşturun.")
            return

        print(f"[Bilgi] Kullanılacak combo5 short dosyası: {combo_short_path}")
        print(f"[Bilgi] Kullanılacak combo5 mid dosyası:   {combo_mid_path}")

        combos_short_df = pd.read_csv(combo_short_path)
        combos_mid_df = pd.read_csv(combo_mid_path)

        if "conditions" not in combos_short_df.columns or "conditions" not in combos_mid_df.columns:
            print("[Hata] COMBO_MINED5 dosyalarında 'conditions' kolonu yok.")
            return
        
        # Backtest verisini yükleyip combolan değerlendir
        back_path = detect_latest_backtest_csv(BASE_DIR)
        if not back_path:
            print("[Uyarı] Backtest dosyası bulunamadı, tüm comboları kullanacağız.")
            # Tüm comboları kullan (değerlendirme yapmadan)
            best_5_low = combos_short_df.head(20)
            best_5_high = combos_short_df.head(20)
            best_15_low = combos_mid_df.head(20) if not combos_mid_df.empty else pd.DataFrame()
        else:
            # Backtest ile değerlendir
            df_backtest = pd.read_csv(back_path)
            
            combos_all_df = pd.concat(
                [
                    combos_short_df.assign(target="short"),
                    combos_mid_df.assign(target="mid15"),
                ],
                ignore_index=True,
            )

            df_stats_all = compute_combo_stats(df_backtest, combos_all_df)
            if df_stats_all.empty:
                print("[Uyarı] Combo istatistikleri hesaplanamadı, tüm comboları kullanacağız.")
                best_5_low = combos_short_df.head(20)
                best_5_high = combos_short_df.head(20)
                best_15_low = combos_mid_df.head(20) if not combos_mid_df.empty else pd.DataFrame()
            else:
                best_5_low, best_5_high, best_15_low = select_best_combos(df_stats_all)

        # Seçilen en iyi kuralları kaydet
        if not best_5_low.empty:
            best_5_low.to_csv(best_5_low_path, index=False, encoding="utf-8-sig")
            print(f"[ÇIKTI] {best_5_low_path}")
        if not best_5_high.empty:
            best_5_high.to_csv(best_5_high_path, index=False, encoding="utf-8-sig")
            print(f"[ÇIKTI] {best_5_high_path}")
        if not best_15_low.empty:
            best_15_low.to_csv(best_15_low_path, index=False, encoding="utf-8-sig")
            print(f"[ÇIKTI] {best_15_low_path}")

    # 4. En iyi comboları BUGÜNÜN verisine uygula
    print(f"[Bilgi] En iyi combolar bugünün multim4 verisine uygulanıyor...")
    apply_best_combos_to_backtest(
        df_selection,
        best_5_low,
        best_5_high,
        best_15_low,
        selection_date_str,
        BASE_DIR,
    )

    print("[Tamamlandı] v5 seçim: 3 yeni tarama listesi + birleşik riskli liste üretildi.")


# ============================================================
#  Argparse ve main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "multim4 -> GERCEK_BACKTEST5_TO_LATEST + COMBO_MINED5_SHORT/MID_15GUN + "
            "v5 seçim listeleri (backtest3 + backtest4 + PatternScore + HitRate5 filtresi)."
        )
    )
    ap.add_argument("--dir", default=None, help="multim4 CSV klasörü (varsayılan: script klasörü)")
    ap.add_argument("--run-backtest", action="store_true", help="Sadece backtest + mining (v5) çalıştır")
    ap.add_argument("--run-selection", action="store_true", help="Sadece mevcut GERCEK_BACKTEST5 + combo5 ile seçim çalıştır")
    ap.add_argument("--all", action="store_true", help="Önce backtest+combo, sonra seçim (tam akış)")

    # backtest3 parametreleri
    ap.add_argument("--top-quantile", type=float, default=None)
    ap.add_argument("--top-quantile-mid", type=float, default=None)
    ap.add_argument("--combo-min-n-single", type=int, default=None)
    ap.add_argument("--combo-min-n-pair", type=int, default=None)
    ap.add_argument("--combo-topk-single", type=int, default=None)
    ap.add_argument("--combo-topk-pair", type=int, default=None)
    ap.add_argument("--num-bins", type=int, default=None)
    ap.add_argument("--max-cat-unique", type=int, default=None)
    ap.add_argument("--last-n-files", type=int, default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--mid-horizon-days", type=int, default=None)
    ap.add_argument("--no-xlsx", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


def merge_config(args):
    cfg = DEFAULT_CONFIG.copy()
    if args.dir:
        cfg["dir"] = args.dir
    if args.top_quantile is not None:
        cfg["top_quantile"] = args.top_quantile
    if args.top_quantile_mid is not None:
        cfg["top_quantile_mid"] = args.top_quantile_mid
    if args.combo_min_n_single is not None:
        cfg["combo_min_n_single"] = args.combo_min_n_single
    if args.combo_min_n_pair is not None:
        cfg["combo_min_n_pair"] = args.combo_min_n_pair
    if args.combo_topk_single is not None:
        cfg["combo_topk_single"] = args.combo_topk_single
    if args.combo_topk_pair is not None:
        cfg["combo_topk_pair"] = args.combo_topk_pair
    if args.num_bins is not None:
        cfg["num_bins"] = args.num_bins
    if args.max_cat_unique is not None:
        cfg["max_cat_unique"] = args.max_cat_unique
    if args.last_n_files is not None:
        cfg["last_n_files"] = args.last_n_files
    if args.start_date is not None:
        cfg["start_date"] = args.start_date
    if args.end_date is not None:
        cfg["end_date"] = args.end_date
    if args.mid_horizon_days is not None:
        cfg["mid_horizon_days"] = args.mid_horizon_days
    if args.no_xlsx:
        cfg["export_xlsx"] = False
    if args.verbose:
        cfg["verbose"] = True
    return cfg


def main():
    args = parse_args()
    cfg = merge_config(args)

    if not (args.run_backtest or args.run_selection or args.all):
        # Varsayılan: tam akış
        args.all = True

    print("[Bilgi] Kullanılan konfig:", cfg)

    if args.run_backtest or args.all:
        run_backtest_v5(cfg)

    if args.run_selection or args.all:
        run_selection_v5_from_backtest()


if __name__ == "__main__":
    main()