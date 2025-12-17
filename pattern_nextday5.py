import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
#  AMAÇ:
#  - GERCEK_BACKTEST5_TO_LATEST_* içinden GERÇEKTEN ÇOK YÜKSELEN
#    günleri seç.
#  - O günlerdeki multim4 teknik pattern'ini öğren.
#  - Bugünkü multim4'te aynı hisselerde bu pattern'e BENZEYEN
#    durumları yakala (hisse bazlı pattern).
#  - Ayrıca BEST_COMBOS5_* global kurallarını bugüne uygula
#    (global pattern).
#  - Sonuç: HisseBazli + Global pattern'e uyan, son 5 günde
#    %20'den fazla koşmamış hisselerin listesi.
#  - Aynı Excel içinde ek sheet: TOP5_DipTeknik (teknik en iyi 5).
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKTEST5_PATTERN = r"^GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.(csv|xlsx)$"
MULTIM4_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
NEXTDAY_RISK_PATTERN = r"^RISK_AYARLI_SECIM_ALL_nextday5_(\d{4}-\d{2}-\d{2})\.xlsx$"

OUT_CSV_TEMPLATE = "PATTERN_NEXTDAY5_{date}.csv"
OUT_XLSX_TEMPLATE = "PATTERN_NEXTDAY5_{date}.xlsx"

COL_SYMBOL = "Sembol"

# İsteğe bağlı: sadece bu evrende kalanları göstermek için
USE_RISK_AYARLI_UNIVERSE = False  # İstersen True yapabilirsin


# ============================================================
#  GENEL YARDIMCI FONKSİYONLAR
# ============================================================

def _detect_latest_by_pattern(folder: str, pattern: str) -> Optional[Tuple[str, str]]:
    rx = re.compile(pattern, re.IGNORECASE)
    candidates: List[Tuple[datetime, str]] = []
    for fname in os.listdir(folder):
        m = rx.match(fname)
        if m:
            d_str = m.group(1)
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            candidates.append((d, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    d, fname = candidates[-1]
    return d.strftime("%Y-%m-%d"), os.path.join(folder, fname)


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1254", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _read_backtest_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".csv":
        return _read_csv_any(path)
    elif ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError:
            raise RuntimeError("openpyxl yüklü değil, backtest5 Excel okunamadı")
    else:
        raise ValueError(f"Desteklenmeyen backtest uzantısı: {ext}")


def _load_multim4_for_date(date_str: str) -> Optional[pd.DataFrame]:
    fname = f"multim4_{date_str}.csv"
    path = os.path.join(BASE_DIR, fname)
    if not os.path.isfile(path):
        return None
    try:
        return _read_csv_any(path)
    except Exception:
        return None


def _detect_latest_multim4() -> Optional[Tuple[str, str]]:
    return _detect_latest_by_pattern(BASE_DIR, MULTIM4_PATTERN)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object and c != COL_SYMBOL:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df


def _ensure_symbol_upper(df: pd.DataFrame, col: str = COL_SYMBOL) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def _compute_last_5d_return(df_multi_latest: pd.DataFrame, latest_date: str) -> pd.Series:
    """
    Son 5 iş günündeki multim4_* dosyalarından Fiyat (Son) kullanarak
    yaklaşık Son_5Gun_% hesaplar.
    Eğer 5 tam gün yoksa, eldeki en eski ile kıyaslar.
    """
    df_latest = df_multi_latest.copy()
    df_latest = _ensure_symbol_upper(df_latest, COL_SYMBOL)

    try:
        d0 = datetime.strptime(latest_date, "%Y-%m-%d").date()
    except Exception:
        return pd.Series(np.nan, index=df_latest.index)

    if "Fiyat (Son)" not in df_latest.columns:
        return pd.Series(np.nan, index=df_latest.index)

    fiyat_son = pd.to_numeric(df_latest["Fiyat (Son)"], errors="coerce")
    prices_hist: Dict[str, float] = dict(zip(df_latest[COL_SYMBOL], fiyat_son))

    prices_5ago: Dict[str, float] = {}
    for offset in range(5, 8):
        d_back = d0 - timedelta(days=offset)
        back_str = d_back.strftime("%Y-%m-%d")
        df_old = _load_multim4_for_date(back_str)
        if df_old is not None and not df_old.empty and COL_SYMBOL in df_old.columns:
            df_old = _ensure_symbol_upper(df_old, COL_SYMBOL)
            if "Fiyat (Son)" in df_old.columns:
                fiyat_old = pd.to_numeric(df_old["Fiyat (Son)"], errors="coerce")
                prices_5ago = dict(zip(df_old[COL_SYMBOL], fiyat_old))
                break

    vals: List[float] = []
    for sym in df_latest[COL_SYMBOL]:
        p0 = prices_hist.get(sym, np.nan)
        p_old = prices_5ago.get(sym, np.nan)
        if np.isnan(p0) or np.isnan(p_old) or p_old == 0:
            vals.append(np.nan)
        else:
            vals.append((p0 - p_old) / p_old * 100.0)

    return pd.Series(vals, index=df_latest.index)


# ============================================================
#  HISSE BAZLI PATTERN (backtest5'ten öğren)
# ============================================================

# "GERÇEKTEN ÇOK YÜKSELEN" günler için eşikler
HISSE_MIN_MAX5 = 25.0
HISSE_MIN_MAX15 = 40.0
HISSE_MAX_DD5 = 35.0
HISSE_MAX_DD15 = 45.0

HISSE_TECH_COLS = [
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Positive",
    "EMA20_gt_EMA50_gt_EMA200",
    "Dipten Uzaklık (%)",
    "ATH'ye Uzaklık (%)",
    "BB_W",
    "PUANLAMA_V4",
    "FinalSkorEx",
    "OBVSkor",
    "VolumeDeltaUp_Sort",
]


def _get_strong_success_rows(df_back: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest5 içinden "en çok yükselen" günleri seç.
    """
    df = df_back.copy()
    df = _ensure_symbol_upper(df, COL_SYMBOL)

    for c in [
        "Max_Getiri_%", "Zirve_Kayip_%",
        "Max_Getiri_15Gun_%", "Zirve_Kayip_15Gun_%",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cond5 = (
        df["Max_Getiri_%"].fillna(-999) >= HISSE_MIN_MAX5
    ) & (df["Zirve_Kayip_%"].fillna(999) <= HISSE_MAX_DD5)

    if "Max_Getiri_15Gun_%" in df.columns and "Zirve_Kayip_15Gun_%" in df.columns:
        cond15 = (
            df["Max_Getiri_15Gun_%"].fillna(-999) >= HISSE_MIN_MAX15
        ) & (df["Zirve_Kayip_15Gun_%"].fillna(999) <= HISSE_MAX_DD15)
    else:
        cond15 = pd.Series(False, index=df.index)

    df["Strong_5g"] = cond5.astype(int)
    df["Strong_15g"] = cond15.astype(int)

    df_good = df[(cond5 | cond15)].copy()
    return df_good


def _build_hisse_pattern_scores(
    df_back: pd.DataFrame,
    df_multi_latest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her hisse için:
      - Backtest'te "çok yükselen" günlerinin multim4 teknik snapshot'ları
      - Bugünkü multim4 snapshot'ı ile benzerlik
      - HisseBazli_Pattern_Skoru, HisseBazli_Beklenen_5g, HisseBazli_Beklenen_15g
      - HisseBazli_Ref_Tarih, HisseBazli_Ref_Max_5g, HisseBazli_Ref_Max_15g
    """
    df_latest = df_multi_latest.copy()
    df_latest = _ensure_symbol_upper(df_latest, COL_SYMBOL)
    df_latest = _coerce_numeric(df_latest)

    df_good = _get_strong_success_rows(df_back)
    if df_good.empty:
        print("[Bilgi] Hisse pattern: backtest içinde 'çok yükselen' gün yok.")
        return pd.DataFrame(columns=[
            COL_SYMBOL,
            "HisseBazli_Pattern_Skoru",
            "HisseBazli_Beklenen_5g", "HisseBazli_Beklenen_15g",
            "HisseBazli_Ref_Tarih", "HisseBazli_Ref_Max_5g", "HisseBazli_Ref_Max_15g",
        ])

    multim_cache: Dict[str, pd.DataFrame] = {}
    pattern_rows: List[Dict[str, Any]] = []

    for _, row in df_good.iterrows():
        sym = row.get(COL_SYMBOL)
        t_str = row.get("Analiz_Tarihi_str")
        if not sym or not t_str or not isinstance(t_str, str):
            continue
        t_str = t_str.strip()
        if not t_str:
            continue

        if t_str not in multim_cache:
            df_m = _load_multim4_for_date(t_str)
            if df_m is None or df_m.empty or COL_SYMBOL not in df_m.columns:
                multim_cache[t_str] = pd.DataFrame()
            else:
                df_m = _ensure_symbol_upper(df_m, COL_SYMBOL)
                df_m = _coerce_numeric(df_m)
                multim_cache[t_str] = df_m

        df_m_t = multim_cache.get(t_str, pd.DataFrame())
        if df_m_t.empty:
            continue

        sub = df_m_t[df_m_t[COL_SYMBOL] == sym]
        if sub.empty:
            continue

        snap = sub.iloc[0]

        rec: Dict[str, Any] = {
            COL_SYMBOL: sym,
            "Analiz_Tarihi_str": t_str,
            "Realized_Max_5g": row.get("Max_Getiri_%", np.nan),
            "Realized_Max_15g": row.get("Max_Getiri_15Gun_%", np.nan),
            "Strong_5g": row.get("Strong_5g", 0),
            "Strong_15g": row.get("Strong_15g", 0),
        }

        for c in HISSE_TECH_COLS:
            rec[f"Tech_{c}"] = snap.get(c, np.nan)

        pattern_rows.append(rec)

    if not pattern_rows:
        print("[Bilgi] Hisse pattern: 'çok yükselen' günlerin multim snapshot'ı yok.")
        return pd.DataFrame(columns=[
            COL_SYMBOL,
            "HisseBazli_Pattern_Skoru",
            "HisseBazli_Beklenen_5g", "HisseBazli_Beklenen_15g",
            "HisseBazli_Ref_Tarih", "HisseBazli_Ref_Max_5g", "HisseBazli_Ref_Max_15g",
        ])

    df_patterns = pd.DataFrame(pattern_rows)
    df_patterns = _coerce_numeric(df_patterns)

    out_rows: List[Dict[str, Any]] = []

    latest_map = {sym: row for sym, row in df_latest.set_index(COL_SYMBOL).iterrows()}

    for sym, grp in df_patterns.groupby(COL_SYMBOL):
        latest_row = latest_map.get(sym)
        if latest_row is None:
            continue

        best_sim = -1.0
        best_bek5 = np.nan
        best_bek15 = np.nan
        best_ref_date = None
        best_ref_max5 = np.nan
        best_ref_max15 = np.nan

        for _, pref in grp.iterrows():
            match_cnt = 0
            total_cnt = 0
            for c in HISSE_TECH_COLS:
                past_val = pref.get(f"Tech_{c}", np.nan)
                now_val = latest_row.get(c, np.nan)
                if pd.isna(past_val) or pd.isna(now_val):
                    continue
                total_cnt += 1

                if c in ("RSI", "Dipten Uzaklık (%)", "ATH'ye Uzaklık (%)", "BB_W"):
                    if abs(float(past_val) - float(now_val)) <= 10.0:
                        match_cnt += 1
                elif c in ("PUANLAMA_V4", "FinalSkorEx", "OBVSkor", "VolumeDeltaUp_Sort", "MACD", "MACD_Signal"):
                    if abs(float(past_val) - float(now_val)) <= 20.0:
                        match_cnt += 1
                elif c in ("MACD_Positive", "EMA20_gt_EMA50_gt_EMA200"):
                    try:
                        if int(past_val) == int(now_val):
                            match_cnt += 1
                    except Exception:
                        pass
                else:
                    if abs(float(past_val) - float(now_val)) <= 15.0:
                        match_cnt += 1

            if total_cnt == 0:
                continue

            sim = match_cnt / total_cnt
            if sim > best_sim:
                best_sim = sim
                best_bek5 = pref.get("Realized_Max_5g", np.nan)
                best_bek15 = pref.get("Realized_Max_15g", np.nan)
                best_ref_date = pref.get("Analiz_Tarihi_str")
                best_ref_max5 = pref.get("Realized_Max_5g", np.nan)
                best_ref_max15 = pref.get("Realized_Max_15g", np.nan)

        if best_sim <= 0:
            continue

        out_rows.append(
            {
                COL_SYMBOL: sym,
                "HisseBazli_Pattern_Skoru": best_sim,
                "HisseBazli_Beklenen_5g": best_bek5,
                "HisseBazli_Beklenen_15g": best_bek15,
                "HisseBazli_Ref_Tarih": best_ref_date,
                "HisseBazli_Ref_Max_5g": best_ref_max5,
                "HisseBazli_Ref_Max_15g": best_ref_max15,
            }
        )

    if not out_rows:
        print("[Bilgi] Hisse pattern: benzerlik skoru >0 olan hisse yok.")
        return pd.DataFrame(columns=[
            COL_SYMBOL,
            "HisseBazli_Pattern_Skoru",
            "HisseBazli_Beklenen_5g", "HisseBazli_Beklenen_15g",
            "HisseBazli_Ref_Tarih", "HisseBazli_Ref_Max_5g", "HisseBazli_Ref_Max_15g",
        ])

    df_hisse = pd.DataFrame(out_rows)
    print(f"[Bilgi] Hisse pattern: skor hesaplanan hisse sayısı = {len(df_hisse)}")
    return df_hisse


# ============================================================
#  GLOBAL PATTERN (BEST_COMBOS5'ten bugüne)
# ============================================================

BEST_5D_LOW_TEMPLATE = "BEST_COMBOS5_5D_LOW_{date}.csv"
BEST_5D_HIGH_TEMPLATE = "BEST_COMBOS5_5D_HIGH_{date}.csv"
BEST_15D_LOW_TEMPLATE = "BEST_COMBOS5_15D_LOW_{date}.csv"  # olmayabilir


def _parse_conditions(cond_str):
    if isinstance(cond_str, list):
        return cond_str
    if not isinstance(cond_str, str):
        return []
    cond_str = cond_str.strip()
    if not cond_str:
        return []
    import ast
    try:
        parsed = ast.literal_eval(cond_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def _apply_combo_set_to_multim_for_pattern(
    df_multi: pd.DataFrame,
    df_combos: pd.DataFrame,
    horizon: str,
) -> pd.DataFrame:
    """
    today.py'deki mantığın sade versiyonu:
      - min_expected filtresi yok,
      - sadece beklenen getiri ve skor atar.
    """
    if df_combos is None or df_combos.empty:
        return pd.DataFrame()

    df_src = df_multi.copy()

    if horizon in ("5_low", "5_high"):
        col_mean = "Global_Beklenen_5g"
        col_risk = "Global_Risk_5g"
        col_id = "Global_Kural_5g"
        col_combo_skor = "Global_Combo_Skor_5g"
        src_mean_col = "Mean_5"
        src_std_col = "Std_5"
        src_ps_col = "PatternScore5"
    else:
        col_mean = "Global_Beklenen_15g"
        col_risk = "Global_Risk_15g"
        col_id = "Global_Kural_15g"
        col_combo_skor = "Global_Combo_Skor_15g"
        src_mean_col = "Mean_15"
        src_std_col = "Std_15"
        src_ps_col = "PatternScore15"

    df_src[col_mean] = np.nan
    df_src[col_risk] = np.nan
    df_src[col_id] = None
    df_src[col_combo_skor] = np.nan

    df_combos = df_combos.copy()
    if "conditions" not in df_combos.columns:
        return df_src
    df_combos["conditions_parsed"] = df_combos["conditions"].apply(_parse_conditions)
    records = df_combos.to_dict(orient="records")

    best_means: List[Optional[float]] = []
    best_risks: List[Optional[float]] = []
    best_ids: List[Optional[str]] = []
    best_scores: List[Optional[float]] = []

    for _, row in df_src.iterrows():
        best_mean = None
        best_risk = None
        best_id = None
        best_score = None

        for idx, combo in enumerate(records):
            conds = combo.get("conditions_parsed") or []
            if not conds:
                continue

            ok = True
            for cond in conds:
                if not isinstance(cond, dict):
                    ok = False
                    break
                col = cond.get("col")
                op = cond.get("op")
                val = cond.get("val")
                if col not in df_src.columns:
                    ok = False
                    break
                v = row.get(col, np.nan)

                if op == "==":
                    if pd.isna(v) or v != val:
                        ok = False
                        break
                elif op == "in_bin":
                    vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
                    if pd.isna(vv):
                        ok = False
                        break
                    left = val.get("left", -np.inf)
                    right = val.get("right", np.inf)
                    closed = val.get("closed", "right")
                    if closed == "right":
                        if not (vv > left and vv <= right):
                            ok = False
                            break
                    else:
                        if not (vv >= left and vv < right):
                            ok = False
                            break
                else:
                    ok = False
                    break

            if not ok:
                continue

            m = combo.get(src_mean_col, None)
            r = combo.get(src_std_col, None)
            s = combo.get(src_ps_col, None)
            if m is None:
                continue

            if (
                best_mean is None
                or (m > best_mean)
                or (m == best_mean and s is not None and best_score is not None and s > best_score)
            ):
                best_mean = m
                best_risk = r
                best_id = f"combo_{idx}"
                best_score = s

        best_means.append(best_mean)
        best_risks.append(best_risk)
        best_ids.append(best_id)
        best_scores.append(best_score)

    df_src[col_mean] = best_means
    df_src[col_risk] = best_risks
    df_src[col_id] = best_ids
    df_src[col_combo_skor] = best_scores

    return df_src


def _build_global_pattern_scores(
    df_multi_latest: pd.DataFrame,
    backtest_date: str,
) -> pd.DataFrame:
    """
    BEST_COMBOS5_* dosyalarındaki global kuralları bugünkü multim4'e uygular,
    Global_Beklenen_5g / 15g ve skorları üretir.
    """
    base_dir = BASE_DIR

    best_5_low_path = os.path.join(base_dir, BEST_5D_LOW_TEMPLATE.format(date=backtest_date))
    best_5_high_path = os.path.join(base_dir, BEST_5D_HIGH_TEMPLATE.format(date=backtest_date))
    best_15_low_path = os.path.join(base_dir, BEST_15D_LOW_TEMPLATE.format(date=backtest_date))

    if not os.path.isfile(best_5_low_path) or not os.path.isfile(best_5_high_path):
        print("[Uyarı] Global pattern: BEST_COMBOS5 5D_LOW/HIGH yok, global skor olmayacak.")
        return pd.DataFrame(columns=[COL_SYMBOL, "Global_Beklenen_5g", "Global_Beklenen_15g",
                                     "Global_Combo_Skor_5g", "Global_Combo_Skor_15g"])

    df_5_low = _read_csv_any(best_5_low_path)
    df_5_high = _read_csv_any(best_5_high_path)

    if os.path.isfile(best_15_low_path):
        df_15_low = _read_csv_any(best_15_low_path)
    else:
        df_15_low = pd.DataFrame()

    for dfc in (df_5_low, df_5_high, df_15_low):
        if dfc is not None and not dfc.empty and "conditions" in dfc.columns:
            dfc["conditions"] = dfc["conditions"].astype(str)

    df_multi = df_multi_latest.copy()
    df_multi = _ensure_symbol_upper(df_multi, COL_SYMBOL)
    df_multi = _coerce_numeric(df_multi)

    print(f"[Bilgi] Global pattern: 5D_LOW kural sayısı = {len(df_5_low)}, 5D_HIGH kural sayısı = {len(df_5_high)}")

    df_g5l = _apply_combo_set_to_multim_for_pattern(df_multi, df_5_low, "5_low")
    df_g5h = _apply_combo_set_to_multim_for_pattern(df_multi, df_5_high, "5_high")
    if df_15_low is not None and not df_15_low.empty:
        print(f"[Bilgi] Global pattern: 15D_LOW kural sayısı = {len(df_15_low)}")
        df_g15 = _apply_combo_set_to_multim_for_pattern(df_multi, df_15_low, "15_low")
    else:
        df_g15 = df_multi.copy()
        df_g15["Global_Beklenen_15g"] = np.nan
        df_g15["Global_Risk_15g"] = np.nan
        df_g15["Global_Kural_15g"] = None
        df_g15["Global_Combo_Skor_15g"] = np.nan

    keep_cols = [COL_SYMBOL]
    merge = df_multi[keep_cols].copy()

    for src in [df_g5l, df_g5h, df_g15]:
        if src is None or src.empty:
            continue
        cols_merge = [c for c in src.columns if c.startswith("Global_")]
        cols_merge = [COL_SYMBOL] + cols_merge
        tmp = src[cols_merge].copy()
        merge = merge.merge(tmp, on=COL_SYMBOL, how="left", suffixes=("", "_dup"))

        for c in list(merge.columns):
            if c.endswith("_dup"):
                base = c[:-4]
                if base in merge.columns:
                    mask = merge[base].isna()
                    merge.loc[mask, base] = merge.loc[mask, c]
                merge = merge.drop(columns=[c])

    merge["Global_Beklenen_5g"] = merge[
        [c for c in merge.columns if c.startswith("Global_Beklenen_5g")]
    ].max(axis=1, skipna=True)

    merge["Global_Combo_Skor_5g"] = merge[
        [c for c in merge.columns if c.startswith("Global_Combo_Skor_5g")]
    ].max(axis=1, skipna=True)

    if "Global_Beklenen_15g" not in merge.columns:
        merge["Global_Beklenen_15g"] = np.nan
    if "Global_Combo_Skor_15g" not in merge.columns:
        merge["Global_Combo_Skor_15g"] = np.nan

    cols_keep_final = [
        COL_SYMBOL,
        "Global_Beklenen_5g",
        "Global_Beklenen_15g",
        "Global_Combo_Skor_5g",
        "Global_Combo_Skor_15g",
    ]
    cols_keep_final = [c for c in cols_keep_final if c in merge.columns]

    print(f"[Bilgi] Global pattern: skor üretilen hisse sayısı = {merge[COL_SYMBOL].nunique()}")
    return merge[cols_keep_final].copy()


# ============================================================
#  ANA AKIŞ
# ============================================================

def main():
    base_dir = BASE_DIR

    # 1) backtest5
    back_info = _detect_latest_by_pattern(base_dir, BACKTEST5_PATTERN)
    if not back_info:
        print("[Hata] PATTERN_NEXTDAY5: GERCEK_BACKTEST5_TO_LATEST_*.csv/xlsx bulunamadı")
        return
    back_date, back_path = back_info
    print(f"[Bilgi] PATTERN_NEXTDAY5: Backtest5 dosyası: {back_path}")

    df_back = _read_backtest_any(back_path)
    if df_back is None or df_back.empty or COL_SYMBOL not in df_back.columns:
        print("[Hata] PATTERN_NEXTDAY5: backtest5 okunamadı veya 'Sembol' yok")
        return
    df_back = _ensure_symbol_upper(df_back, COL_SYMBOL)

    # 2) bugün multim4
    multim_info = _detect_latest_multim4()
    if not multim_info:
        print("[Hata] PATTERN_NEXTDAY5: multim4_*.csv bulunamadı")
        return
    multim_date, multim_path = multim_info
    print(f"[Bilgi] PATTERN_NEXTDAY5: multim4 dosyası: {multim_path}")

    df_multi = _read_csv_any(multim_path)
    if df_multi is None or df_multi.empty or COL_SYMBOL not in df_multi.columns:
        print("[Hata] PATTERN_NEXTDAY5: multim4 okunamadı veya 'Sembol' yok")
        return
    df_multi = _ensure_symbol_upper(df_multi, COL_SYMBOL)
    df_multi = _coerce_numeric(df_multi)

    # 3) Hisse bazlı pattern skorları
    print("[Bilgi] PATTERN_NEXTDAY5: Hisse bazlı pattern skorları hesaplanıyor...")
    df_hisse_pattern = _build_hisse_pattern_scores(df_back, df_multi)

    # 4) Global pattern skorları
    print("[Bilgi] PATTERN_NEXTDAY5: Global pattern skorları hesaplanıyor...")
    df_global_pattern = _build_global_pattern_scores(df_multi, back_date)

    # 5) Ana tablo: multim4 tüm kolonlar + pattern skorları
    df_all = df_multi.copy()

    if not df_hisse_pattern.empty:
        df_all = df_all.merge(df_hisse_pattern, on=COL_SYMBOL, how="left")
    else:
        print("[Uyarı] PATTERN_NEXTDAY5: Hisse bazlı pattern skoru bulunamadı.")

    if not df_global_pattern.empty:
        df_all = df_all.merge(df_global_pattern, on=COL_SYMBOL, how="left")
    else:
        print("[Uyarı] PATTERN_NEXTDAY5: Global pattern skoru bulunamadı.")

    # 6) Son gün %
    son_gun_col = None
    for c in ["Son Gün %", "Son_Gün_%", "Son_Gun_%"]:
        if c in df_all.columns:
            son_gun_col = c
            break

    if son_gun_col is None:
        df_all["Son_Gun_%"] = 0.0
    else:
        df_all["Son_Gun_%"] = pd.to_numeric(df_all[son_gun_col], errors="coerce").fillna(0.0)

    # 7) Son_5Gun_%
    print("[Bilgi] PATTERN_NEXTDAY5: Son_5Gun_% hesaplanıyor...")
    df_all["Son_5Gun_%"] = _compute_last_5d_return(df_all, multim_date)

    # 8) Beklenen getiriler ve pattern skorlar
    for c in [
        "HisseBazli_Pattern_Skoru",
        "HisseBazli_Beklenen_5g", "HisseBazli_Beklenen_15g",
        "Global_Beklenen_5g", "Global_Beklenen_15g",
        "Global_Combo_Skor_5g", "Global_Combo_Skor_15g",
    ]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    df_all["Beklenen_5g_Toplam"] = df_all[[
        c for c in ["HisseBazli_Beklenen_5g", "Global_Beklenen_5g"] if c in df_all.columns
    ]].max(axis=1, skipna=True)

    df_all["Beklenen_15g_Toplam"] = df_all[[
        c for c in ["HisseBazli_Beklenen_15g", "Global_Beklenen_15g"] if c in df_all.columns
    ]].max(axis=1, skipna=True)

    df_all["Ana_Beklenen"] = df_all[["Beklenen_5g_Toplam", "Beklenen_15g_Toplam"]].max(axis=1, skipna=True)

    # 9) Opsiyonel: RISK_AYARLI evreni ile kesiştir
    if USE_RISK_AYARLI_UNIVERSE:
        ref_info = _detect_latest_by_pattern(base_dir, NEXTDAY_RISK_PATTERN)
        if ref_info:
            _, ref_path = ref_info
            print(f"[Bilgi] PATTERN_NEXTDAY5: RISK_AYARLI referans dosya: {ref_path}")
            try:
                ref_df = pd.read_excel(ref_path, engine="openpyxl")
                if "Sembol" in ref_df.columns:
                    ref_df = _ensure_symbol_upper(ref_df, COL_SYMBOL)
                    ref_syms = set(ref_df[COL_SYMBOL].dropna().astype(str).tolist())
                    before_ref = len(df_all)
                    df_all = df_all[df_all[COL_SYMBOL].isin(ref_syms)].copy()
                    after_ref = len(df_all)
                    print(f"[Bilgi] PATTERN_NEXTDAY5: RISK_AYARLI evren filtresi: {before_ref} -> {after_ref}")
            except Exception as e:
                print(f"[Uyarı] RISK_AYARLI dosya okunamadı: {e}")

    # 10) Pattern filtresi: Sadece gerçekten pattern'e uyanlar
    hisse_thr = 0.65
    global_5_thr = 10.0
    global_15_thr = 15.0  # 15g için eşik

    cond_hisse = df_all.get("HisseBazli_Pattern_Skoru", pd.Series(0, index=df_all.index)).fillna(0) >= hisse_thr
    cond_global5 = df_all.get("Global_Beklenen_5g", pd.Series(np.nan, index=df_all.index)).fillna(-999) >= global_5_thr
    cond_global15 = df_all.get("Global_Beklenen_15g", pd.Series(np.nan, index=df_all.index)).fillna(-999) >= global_15_thr

    df_all["Kaynak_HisseBazli"] = cond_hisse.astype(int)
    df_all["Kaynak_Global"] = (cond_global5 | cond_global15).astype(int)

    mask_pattern = cond_hisse | cond_global5 | cond_global15

    before_pat = len(df_all)
    df_all = df_all[mask_pattern].copy()
    after_pat = len(df_all)
    print(f"[Bilgi] PATTERN_NEXTDAY5: Pattern filtresi (hisse>={hisse_thr} veya 5g>={global_5_thr} veya 15g>={global_15_thr}): {before_pat} -> {after_pat} satır")

    if df_all.empty:
        print("[Uyarı] PATTERN_NEXTDAY5: Pattern filtresinden sonra hisse kalmadı")
        return

    # 11) Overextended filtresi: Son_5Gun_% > 20 elenecek
    df_all["Son_5Gun_%"] = pd.to_numeric(df_all["Son_5Gun_%"], errors="coerce")
    before_ov = len(df_all)
    df_all = df_all[(df_all["Son_5Gun_%"].isna()) | (df_all["Son_5Gun_%"] <= 20.0)].copy()
    after_ov = len(df_all)
    print(f"[Bilgi] PATTERN_NEXTDAY5: Son_5Gun_% filtresi (<=20): {before_ov} -> {after_ov} satır")

    if df_all.empty:
        print("[Uyarı] PATTERN_NEXTDAY5: Son_5Gun_% filtresinden sonra hisse kalmadı")
        return

    # 12) Teknik kalite skorları (bugünkü multim'e göre)
    def _teknik_skor_satir(row: pd.Series) -> float:
        val = 0.0
        p4 = row.get("PUANLAMA_V4", np.nan)
        fex = row.get("FinalSkorEx", np.nan)
        rsi = row.get("RSI", np.nan)

        try:
            fex = float(fex)
            if not np.isnan(fex):
                if fex > 0:
                    val += np.clip(fex / 8.0, 0.0, 4.0)
                else:
                    val += np.clip(fex / 15.0, -3.0, 0.0)
        except Exception:
            pass

        try:
            p4 = float(p4)
            if not np.isnan(p4):
                val += np.clip(p4 / 3.0, -2.0, 3.0)
        except Exception:
            pass

        try:
            rsi = float(rsi)
            if not np.isnan(rsi):
                if 40 <= rsi <= 60:
                    val += 2.0
                elif 30 <= rsi < 40 or 60 < rsi <= 75:
                    val += 1.0
                elif rsi > 80:
                    val -= 2.0
                elif rsi < 25:
                    val -= 1.0
        except Exception:
            pass

        return float(val)

    df_all["Teknik_Skor_Basit"] = df_all.apply(_teknik_skor_satir, axis=1)

    # 13) Sıralama: teknik en iyiden aşağı
    df_all["FinalSkorEx"]       = pd.to_numeric(df_all.get("FinalSkorEx", 0), errors="coerce").fillna(0.0)
    df_all["PUANLAMA_V4"]       = pd.to_numeric(df_all.get("PUANLAMA_V4", 0), errors="coerce").fillna(0.0)
    df_all["RSI"]               = pd.to_numeric(df_all.get("RSI", 0), errors="coerce").fillna(0.0)
    df_all["Teknik_Skor_Basit"] = pd.to_numeric(df_all["Teknik_Skor_Basit"], errors="coerce").fillna(-9999.0)
    df_all["Son_5Gun_%"]        = pd.to_numeric(df_all["Son_5Gun_%"], errors="coerce").fillna(0.0)

    df_all = df_all.sort_values(
        by=["FinalSkorEx", "PUANLAMA_V4", "Teknik_Skor_Basit", "RSI", "Son_5Gun_%"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    # 14) Numerik kolonları yuvarla
    num_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    if num_cols:
        df_all[num_cols] = df_all[num_cols].round(2)

    # ---------------- TOP5_DipTeknik: ana listeden en iyi 5 teknik ----------------
    df_top5 = pd.DataFrame()
    if not df_all.empty:
        df_top5 = df_all.copy()

        # Çok çökmesin: Son_5Gun_% >= -25 yeterli
        df_top5["Son_5Gun_%"] = pd.to_numeric(df_top5["Son_5Gun_%"], errors="coerce")
        mask_5g = (df_top5["Son_5Gun_%"].isna()) | (df_top5["Son_5Gun_%"] >= -25.0)
        df_top5 = df_top5[mask_5g].copy()

        if not df_top5.empty:
            df_top5 = df_top5.sort_values(
                by=["FinalSkorEx", "PUANLAMA_V4", "Teknik_Skor_Basit", "Son_5Gun_%"],
                ascending=[False, False, False, True],
            ).head(5).reset_index(drop=True)

    # 15) Çıktılar
    out_csv = os.path.join(base_dir, OUT_CSV_TEMPLATE.format(date=multim_date))
    out_xlsx = os.path.join(base_dir, OUT_XLSX_TEMPLATE.format(date=multim_date))

    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ÇIKTI] PATTERN_NEXTDAY5 CSV: {out_csv}")

    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, index=False, sheet_name="PATTERN_NEXTDAY5")
            if df_top5 is not None and not df_top5.empty:
                df_top5.to_excel(writer, index=False, sheet_name="TOP5_DipTeknik")
            else:
                print("[Bilgi] TOP5_DipTeknik: filtre sonrası satır kalmadı, sheet yazılmadı.")
        print(f"[ÇIKTI] PATTERN_NEXTDAY5 XLSX: {out_xlsx}")
    except ModuleNotFoundError:
        print("[Uyarı] PATTERN_NEXTDAY5: openpyxl yok, Excel yazılamadı; sadece CSV var.")


if __name__ == "__main__":
    main()