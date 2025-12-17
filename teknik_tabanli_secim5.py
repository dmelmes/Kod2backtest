import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKTEST5_PATTERN = r"^GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.csv$"
MULTIM4_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"

OUT_TEMPLATE = "TEKNIK_TABANLI_SECIM_nextday5_{date}.xlsx"


def _detect_latest_by_pattern(folder: str, pattern: str) -> Optional[tuple[str, str]]:
    rx = re.compile(pattern, re.IGNORECASE)
    candidates: List[tuple[datetime, str]] = []
    for fname in os.listdir(folder):
        m = rx.match(fname)
        if m:
            d_str = m.group(1)
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
            except Exception:
                continue
            candidates.append((d, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    d, fname = candidates[-1]
    return d.strftime("%Y-%m-%d"), os.path.join(folder, fname)


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------
# Teknik skorlar (orijinal ile aynı)
# ---------------------------------------------------------

def teknik_trend_skor(row: pd.Series) -> float:
    skor = 0.0
    if row.get("EMA20_gt_EMA50_gt_EMA200") == 1:
        skor += 3

    macd = row.get("MACD")
    macd_sig = row.get("MACD_Signal")
    macd_pos = row.get("MACD_Positive")
    try:
        macd = float(macd)
        macd_sig = float(macd_sig)
    except (TypeError, ValueError):
        macd = np.nan
        macd_sig = np.nan

    if not np.isnan(macd) and not np.isnan(macd_sig):
        if macd > macd_sig and macd_pos == 1:
            skor += 3
        elif macd < macd_sig and macd_pos == 0:
            skor -= 2

    alpha = row.get("AlphaTrend")
    fiyat = row.get("Fiyat (Son)")
    try:
        alpha = float(alpha)
        fiyat = float(fiyat)
        if not np.isnan(alpha) and not np.isnan(fiyat):
            if alpha < fiyat:
                skor += 1
            else:
                skor -= 1
    except (TypeError, ValueError):
        pass

    dir_1d = str(row.get("EW_Direction_1d") or "").lower()
    if dir_1d == "up":
        skor += 1
    elif dir_1d == "down":
        skor -= 1

    return skor


def teknik_momentum_skor(row: pd.Series) -> float:
    skor = 0.0
    rsi = row.get("RSI")
    adx = row.get("ADX")
    try:
        rsi = float(rsi)
    except (TypeError, ValueError):
        rsi = np.nan
    try:
        adx = float(adx)
    except (TypeError, ValueError):
        adx = np.nan

    if not np.isnan(rsi):
        if 40 <= rsi <= 60:
            skor += 2
        elif 30 <= rsi < 40 or 60 < rsi <= 75:
            skor += 1
        elif rsi > 80:
            skor -= 2
        elif rsi < 25:
            skor -= 1

    if not np.isnan(adx):
        if 20 <= adx <= 35:
            skor += 2
        elif adx > 35:
            skor += 1

    obv_skor = row.get("OBVSkor")
    vol_skor = row.get("VolumeDeltaUp_Sort")
    try:
        obv_skor = float(obv_skor)
        skor += np.clip(obv_skor / 20.0, -1.0, 1.0)
    except (TypeError, ValueError):
        pass
    try:
        vol_skor = float(vol_skor)
        if vol_skor > 0:
            skor += 0.5
    except (TypeError, ValueError):
        pass

    return skor


def teknik_risk_skor(row: pd.Series) -> float:
    skor = 0.0

    dip = row.get("Dipten Uzaklık (%)")
    ath = row.get("ATH'ye Uzaklık (%)")
    bb_w = row.get("BB_W")
    try:
        dip = float(dip)
    except (TypeError, ValueError):
        dip = np.nan
    try:
        ath = float(ath)
    except (TypeError, ValueError):
        ath = np.nan
    try:
        bb_w = float(bb_w)
    except (TypeError, ValueError):
        bb_w = np.nan

    if not np.isnan(dip):
        if dip <= 30:
            skor += 2
        elif dip <= 50:
            skor += 1
        elif dip > 85:
            skor -= 2

    if not np.isnan(ath):
        if 8 <= ath <= 60:
            skor += 1
        elif ath < 3:
            skor -= 1
        elif ath > 95:
            skor -= 1

    if not np.isnan(bb_w):
        if bb_w <= 20:
            skor += 1
        elif bb_w > 45:
            skor -= 1

    for c in ["DipSkor", "ATHSkor", "TazeSkor", "BONUS_SCORE"]:
        v = row.get(c)
        try:
            v = float(v)
            skor += np.clip(v / 20.0, -1.0, 1.0)
        except (TypeError, ValueError):
            pass

    return skor


def teknik_sinyal_formasyon_skor(row: pd.Series) -> float:
    skor = 0.0

    sinyal = row.get("Sinyal")
    try:
        sinyal = int(sinyal)
    except (TypeError, ValueError):
        sinyal = 0

    if sinyal == 100:
        skor += 3
    elif sinyal == 50:
        skor += 1
    elif sinyal == -50:
        skor -= 1
    elif sinyal == -100:
        skor -= 2

    if row.get("Confirmed_BUY") == 1:
        skor += 3
    if row.get("Confirmed_SELL") == 1:
        skor -= 3

    if row.get("Cup&Handle Formasyonu") == 1:
        skor += 1.5
    if row.get("Bull Flag Formasyonu") == 1:
        skor += 1.5
    if row.get("TOBO Formasyonu") == 1:
        skor += 2.0
    if row.get("Gelişmiş TOBO") == 1:
        skor += 2.0

    form_sort = row.get("Formasyon_Sort")
    try:
        form_sort = float(form_sort)
        if form_sort > 0:
            skor += 0.5
    except (TypeError, ValueError):
        pass

    return skor


def teknik_toplam_skor(row: pd.Series) -> float:
    return (
        row.get("TrendSkor_T", 0.0)
        + row.get("MomentumSkor_T", 0.0)
        + row.get("RiskSkor_T", 0.0)
        + row.get("SinyalFormasyonSkor_T", 0.0)
    )


def teknik_ana_filtre(row: pd.Series) -> bool:
    """Tekniği çok kötü olanları eleyen hafif filtre."""
    if row.get("EMA20_gt_EMA50_gt_EMA200") != 1:
        return False

    if row.get("MACD_Positive") not in (1, "1", True):
        return False

    rsi = row.get("RSI")
    try:
        rsi = float(rsi)
    except (TypeError, ValueError):
        rsi = np.nan
    if np.isnan(rsi) or not (30 <= rsi <= 80):
        return False

    dip = row.get("Dipten Uzaklık (%)")
    ath = row.get("ATH'ye Uzaklık (%)")
    bb_w = row.get("BB_W")
    try:
        dip = float(dip)
    except (TypeError, ValueError):
        dip = np.nan
    try:
        ath = float(ath)
    except (TypeError, ValueError):
        ath = np.nan
    try:
        bb_w = float(bb_w)
    except (TypeError, ValueError):
        bb_w = np.nan

    if not np.isnan(dip) and dip > 85:
        return False
    if not np.isnan(ath) and (ath < 2 or ath > 95):
        return False
    if not np.isnan(bb_w) and bb_w > 45:
        return False

    sinyal = row.get("Sinyal")
    try:
        sinyal = int(sinyal)
    except (TypeError, ValueError):
        sinyal = 0
    if sinyal <= -100 and row.get("Confirmed_BUY") != 1:
        return False

    return True


# ---------------------------------------------------------
# Backtest5 -> sembol bazlı performans
# ---------------------------------------------------------

def compute_symbol_perf_stats(df_back: pd.DataFrame) -> pd.DataFrame:
    needed_cols = [
        "Sembol",
        "Max_Getiri_%",
        "Zirve_Kayip_%",
        "Max_Getiri_15Gun_%",
        "Zirve_Kayip_15Gun_%",
    ]
    for c in needed_cols:
        if c not in df_back.columns:
            print(f"[Uyarı] Backtest kolon eksik: {c}")
    df = df_back.copy()
    if "Sembol" not in df.columns:
        return pd.DataFrame()

    for c in ["Max_Getiri_%", "Zirve_Kayip_%", "Max_Getiri_15Gun_%", "Zirve_Kayip_15Gun_%"]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    df_valid = df.dropna(subset=["Sembol"])
    if df_valid.empty:
        return pd.DataFrame()

    grouped = df_valid.groupby("Sembol")
    rows: List[Dict[str, Any]] = []

    for sym, g in grouped:
        g5 = g.dropna(subset=["Max_Getiri_%", "Zirve_Kayip_%"])
        g15 = g.dropna(subset=["Max_Getiri_15Gun_%", "Zirve_Kayip_15Gun_%"])

        rec: Dict[str, Any] = {"Sembol": sym}

        if not g5.empty:
            max5 = g5["Max_Getiri_%"]
            dd5 = g5["Zirve_Kayip_%"]
            rec["N_5"] = int(len(max5))
            rec["Ort_Max_Getiri_5g"] = float(max5.mean())
            rec["Ort_Zirve_Kayip_5g"] = float(dd5.mean())
            rec["Skor_5"] = rec["Ort_Max_Getiri_5g"] - 0.7 * rec["Ort_Zirve_Kayip_5g"]
        else:
            rec["N_5"] = 0
            rec["Ort_Max_Getiri_5g"] = np.nan
            rec["Ort_Zirve_Kayip_5g"] = np.nan
            rec["Skor_5"] = np.nan

        if not g15.empty:
            max15 = g15["Max_Getiri_15Gun_%"]
            dd15 = g15["Zirve_Kayip_15Gun_%"]
            rec["N_15"] = int(len(max15))
            rec["Ort_Max_Getiri_15g"] = float(max15.mean())
            rec["Ort_Zirve_Kayip_15g"] = float(dd15.mean())
            rec["Skor_15"] = rec["Ort_Max_Getiri_15g"] - 0.7 * rec["Ort_Zirve_Kayip_15g"]
        else:
            rec["N_15"] = 0
            rec["Ort_Max_Getiri_15g"] = np.nan
            rec["Ort_Zirve_Kayip_15g"] = np.nan
            rec["Skor_15"] = np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Ana akış
# ---------------------------------------------------------

def main():
    base_dir = BASE_DIR

    back_info = _detect_latest_by_pattern(base_dir, BACKTEST5_PATTERN)
    if not back_info:
        print("[Hata] teknik_tabanli_secim5: GERCEK_BACKTEST5_TO_LATEST_*.csv yok.")
        return
    date_str, back_path = back_info
    print(f"[Bilgi] Backtest5 dosyası: {back_path}")

    multi_info = _detect_latest_by_pattern(base_dir, MULTIM4_PATTERN)
    if not multi_info:
        print("[Hata] teknik_tabanli_secim5: multim4_*.csv yok.")
        return
    multi_date, multi_path = multi_info
    print(f"[Bilgi] multim4 dosyası: {multi_path}")

    df_back = pd.read_csv(back_path, low_memory=False)
    df_multi = pd.read_csv(multi_path)
    if "Sembol" not in df_back.columns or "Sembol" not in df_multi.columns:
        print("[Hata] 'Sembol' kolonu eksik.")
        return

    # 1) backtest5 -> performans
    df_perf = compute_symbol_perf_stats(df_back)
    if df_perf.empty:
        print("[Hata] performans istatistiği üretilemedi.")
        return

    # 2) multim4 -> teknik skorlar
    num_cols_multi = [
        "Fiyat (Son)", "MACD", "MACD_Signal", "MACD_Positive",
        "RSI", "ADX", "Dipten Uzaklık (%)", "ATH'ye Uzaklık (%)",
        "BB_W", "PUANLAMA_V4", "FinalSkorEx",
        "DipSkor", "ATHSkor", "OBVSkor", "TazeSkor", "VolumeDeltaUp_Sort",
    ]
    for c in num_cols_multi:
        if c in df_multi.columns:
            df_multi[c] = _to_num(df_multi[c])

    df_multi["TrendSkor_T"] = df_multi.apply(teknik_trend_skor, axis=1)
    df_multi["MomentumSkor_T"] = df_multi.apply(teknik_momentum_skor, axis=1)
    df_multi["RiskSkor_T"] = df_multi.apply(teknik_risk_skor, axis=1)
    df_multi["SinyalFormasyonSkor_T"] = df_multi.apply(teknik_sinyal_formasyon_skor, axis=1)
    df_multi["TeknikSkor_T"] = df_multi.apply(teknik_toplam_skor, axis=1)
    df_multi["TeknikFiltre_AL"] = df_multi.apply(teknik_ana_filtre, axis=1)

    # 3) Merge
    df_merge = df_perf.merge(df_multi, on="Sembol", how="left", suffixes=("", "_m4"))
    df_merge = df_merge[df_merge["TeknikFiltre_AL"] == True].copy()

    # Pozitif risk-ayarlı getiri + yeterli örnek
    df_merge = df_merge[
        ((df_merge["Skor_5"].fillna(-999) > 0) & (df_merge["N_5"] >= 5)) |
        ((df_merge["Skor_15"].fillna(-999) > 0) & (df_merge["N_15"] >= 5))
    ].copy()
    if df_merge.empty:
        print("[Uyarı] teknik_tabanli_secim5: uygun hisse yok.")
        return

    alfa = 0.5
    df_merge["Final_5"] = df_merge["Skor_5"].fillna(-999) + alfa * df_merge["TeknikSkor_T"].fillna(0)
    df_merge["Final_15"] = df_merge["Skor_15"].fillna(-999) + alfa * df_merge["TeknikSkor_T"].fillna(0)
    df_merge["Final_Kombi"] = df_merge[["Skor_5", "Skor_15"]].fillna(-999).max(axis=1) + alfa * df_merge["TeknikSkor_T"].fillna(0)

    # 4) Kolon sırası
    base_cols = [
        "Sembol",
        "Fiyat (Son)",
        "Sinyal",
        "Ort_Max_Getiri_5g", "Ort_Zirve_Kayip_5g", "Skor_5", "N_5",
        "Ort_Max_Getiri_15g", "Ort_Zirve_Kayip_15g", "Skor_15", "N_15",
        "Dipten Uzaklık (%)", "ATH'ye Uzaklık (%)", "BB_W",
        "TeknikSkor_T", "TrendSkor_T", "MomentumSkor_T",
        "RiskSkor_T", "SinyalFormasyonSkor_T",
    ]
    other_cols = [
        c for c in df_merge.columns
        if c not in base_cols + ["Final_5", "Final_15", "Final_Kombi", "TeknikFiltre_AL"]
    ]

    cols_5 = base_cols + other_cols + ["Final_5"]
    cols_15 = base_cols + other_cols + ["Final_15"]
    cols_k = base_cols + other_cols + ["Final_Kombi"]

    df_5 = df_merge.sort_values("Final_5", ascending=False)[cols_5]
    df_15 = df_merge.sort_values("Final_15", ascending=False)[cols_15]
    df_k = df_merge.sort_values("Final_Kombi", ascending=False)[cols_k]

    # 5) İki ek liste: düşük risk ve agresif

    # Düşük risk (dip odaklı, defansif)
    df_low_risk = df_merge.copy()
    df_low_risk = df_low_risk[
        (df_low_risk["Dipten Uzaklık (%)"] <= 50) &
        (df_low_risk["Ort_Zirve_Kayip_5g"] <= 5) &
        (df_low_risk["Skor_5"] > 0)
    ].copy()
    df_low_risk["Final_Dip"] = df_low_risk["Skor_5"] + alfa * df_low_risk["TeknikSkor_T"].fillna(0)
    df_low_risk = df_low_risk.sort_values("Final_Dip", ascending=False)
    cols_dip = base_cols + other_cols + ["Final_Dip"]
    df_low_risk = df_low_risk[cols_dip]

    # Agresif 15g: Skor_15 > 10 ve N_15 >= 10
    df_aggr = df_merge[
        (df_merge["Skor_15"] > 10) &
        (df_merge["N_15"] >= 10)
    ].copy()
    df_aggr["Final_Agresif_15"] = df_aggr["Final_15"]
    df_aggr = df_aggr.sort_values("Final_Agresif_15", ascending=False)
    cols_aggr = base_cols + other_cols + ["Final_Agresif_15"]
    df_aggr = df_aggr[cols_aggr]

    # 6) Sayısal kolonları yuvarla
    def _round(df_in: pd.DataFrame) -> pd.DataFrame:
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if num_cols:
            df_in[num_cols] = df_in[num_cols].round(2)
        return df_in

    df_5 = _round(df_5)
    df_15 = _round(df_15)
    df_k = _round(df_k)
    df_low_risk = _round(df_low_risk)
    df_aggr = _round(df_aggr)

    # 7) Excel'e yaz
    out_path = os.path.join(base_dir, OUT_TEMPLATE.format(date=date_str))
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_5.to_excel(writer, index=False, sheet_name="TEK_5GUN")
            df_15.to_excel(writer, index=False, sheet_name="TEK_15GUN")
            df_k.to_excel(writer, index=False, sheet_name="TEK_KOMBI")
            if not df_low_risk.empty:
                df_low_risk.to_excel(writer, index=False, sheet_name="TEK_DUSUK_RISK")
            if not df_aggr.empty:
                df_aggr.to_excel(writer, index=False, sheet_name="TEK_AGRESIF_15G")
        print(f"[ÇIKTI] teknik_tabanli_secim5: {out_path}")
    except ModuleNotFoundError:
        print("[Uyarı] teknik_tabanli_secim5: openpyxl yok, Excel yazılamadı.")


if __name__ == "__main__":
    main()