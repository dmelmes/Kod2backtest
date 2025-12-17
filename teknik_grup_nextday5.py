import os
import re
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MULTIM4_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
NEXTDAY5_TEMPLATE = "RISK_AYARLI_SECIM_ALL_nextday5_{date}.xlsx"
SHEET_NAME = "NEXTDAY5_TEKNIKGRUP"


def _detect_latest_multim4(folder: str) -> Optional[tuple[str, str]]:
    """Klasördeki en güncel multim4_YYYY-MM-DD.csv dosyasını ve tarihini bul."""
    rx = re.compile(MULTIM4_PATTERN, re.IGNORECASE)
    candidates: list[tuple[date, str]] = []
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


def _safe_get(row, key, default=np.nan):
    if key in row:
        return row[key]
    try:
        return row.get(key, default)
    except Exception:
        return default


# ---------------- skor fonksiyonları ----------------

def hesapla_trend_skor(row: pd.Series) -> float:
    skor = 0.0
    if _safe_get(row, "EMA20_gt_EMA50_gt_EMA200") == 1:
        skor += 3

    macd = _safe_get(row, "MACD")
    macd_sig = _safe_get(row, "MACD_Signal")
    macd_pos = _safe_get(row, "MACD_Positive")
    try:
        macd = float(macd)
        macd_sig = float(macd_sig)
    except (TypeError, ValueError):
        macd = np.nan
        macd_sig = np.nan

    if not np.isnan(macd) and not np.isnan(macd_sig):
        if macd > macd_sig and macd_pos == 1:
            skor += 2
        elif macd < macd_sig and macd_pos == 0:
            skor -= 2

    alpha = _safe_get(row, "AlphaTrend")
    fiyat_guncel = _safe_get(row, "Fiyat_Guncel")
    try:
        alpha = float(alpha)
        fiyat_guncel = float(fiyat_guncel)
        if not np.isnan(alpha) and not np.isnan(fiyat_guncel):
            if alpha < fiyat_guncel:
                skor += 1
            else:
                skor -= 1
    except (TypeError, ValueError):
        pass

    return skor


def hesapla_momentum_skor(row: pd.Series) -> float:
    skor = 0.0
    rsi = _safe_get(row, "RSI")
    try:
        rsi = float(rsi)
    except (TypeError, ValueError):
        rsi = np.nan

    if not np.isnan(rsi):
        if 40 <= rsi <= 60:
            skor += 2
        if 30 <= rsi < 40:
            skor += 1
        if rsi > 70:
            skor -= 2
        if rsi < 30:
            skor -= 1

    adx = _safe_get(row, "ADX")
    try:
        adx = float(adx)
    except (TypeError, ValueError):
        adx = np.nan

    if not np.isnan(adx):
        if 20 <= adx <= 35:
            skor += 2
        elif adx > 35:
            skor += 1

    return skor


def hesapla_risk_skor(row: pd.Series) -> float:
    skor = 0.0
    dip_uzak = _safe_get(row, "Dipten Uzaklık (%)")
    ath_uzak = _safe_get(row, "ATH'ye Uzaklık (%)")
    bb_w = _safe_get(row, "BB_W")

    try:
        dip_uzak = float(dip_uzak)
    except (TypeError, ValueError):
        dip_uzak = np.nan
    try:
        ath_uzak = float(ath_uzak)
    except (TypeError, ValueError):
        ath_uzak = np.nan
    try:
        bb_w = float(bb_w)
    except (TypeError, ValueError):
        bb_w = np.nan

    if not np.isnan(dip_uzak):
        if dip_uzak <= 25:
            skor += 2
        elif dip_uzak <= 40:
            skor += 1
        elif dip_uzak > 60:
            skor -= 2

    if not np.isnan(ath_uzak):
        if 10 <= ath_uzak <= 40:
            skor += 1
        elif ath_uzak < 5:
            skor -= 1
        elif ath_uzak > 70:
            skor -= 1

    if not np.isnan(bb_w):
        if bb_w <= 15:
            skor += 1
        elif bb_w > 25:
            skor -= 1

    return skor


def hesapla_sinyal_skor(row: pd.Series) -> float:
    skor = 0.0
    sinyal = _safe_get(row, "Sinyal")
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

    if _safe_get(row, "Confirmed_BUY") == 1:
        skor += 2
    if _safe_get(row, "Confirmed_SELL") == 1:
        skor -= 2

    return skor


def hesapla_formasyon_skor(row: pd.Series) -> float:
    skor = 0.0
    if _safe_get(row, "Cup&Handle Formasyonu") == 1:
        skor += 1
    if _safe_get(row, "Bull Flag Formasyonu") == 1:
        skor += 1
    if _safe_get(row, "TOBO Formasyonu") == 1:
        skor += 2
    if _safe_get(row, "Gelişmiş TOBO") == 1:
        skor += 2
    if _safe_get(row, "Gelişmiş Cup & Handle") == 1:
        skor += 1
    return skor


def hesapla_toplam_teknik_skor(row: pd.Series) -> float:
    skor = 0.0
    skor += _safe_get(row, "TrendSkor", 0.0)
    skor += _safe_get(row, "MomentumSkor", 0.0)
    skor += _safe_get(row, "RiskSkor", 0.0)
    skor += _safe_get(row, "SinyalSkor", 0.0)
    skor += _safe_get(row, "FormasyonSkor", 0.0)

    puan = _safe_get(row, "PUANLAMA_V4")
    try:
        puan = float(puan)
        if puan >= 5:
            skor += 1
    except (TypeError, ValueError):
        pass

    final_ex = _safe_get(row, "FinalSkorEx")
    try:
        final_ex = float(final_ex)
        if final_ex >= 0:
            skor += 1
    except (TypeError, ValueError):
        pass

    return skor


def teknik_grup_etk(row: pd.Series) -> str:
    t = _safe_get(row, "TrendSkor", 0.0)
    r = _safe_get(row, "RiskSkor", 0.0)
    m = _safe_get(row, "MomentumSkor", 0.0)
    s = _safe_get(row, "SinyalSkor", 0.0)

    try:
        t, r, m, s = map(float, [t, r, m, s])
    except (TypeError, ValueError):
        return "TEKNIK_UNKNOWN"

    if t >= 4 and r >= 2 and s >= 2:
        return "TEKNIK_A1_GUCLU_TREND_DUSUK_RISK"
    if t >= 4 and r < 2:
        return "TEKNIK_A2_GUCLU_TREND_YUKSEK_RISK"
    if 2 <= t < 4:
        if m >= 2:
            return "TEKNIK_B1_ORTA_TREND_POTANSIYEL"
        else:
            return "TEKNIK_B2_ORTA_TREND_ZAYIF_MOMENTUM"
    if t < 2 and s <= 0:
        return "TEKNIK_C1_ZAYIF_TREND"
    return "TEKNIK_B3_NOTR"


def fiyat_konum_kategori(row: pd.Series) -> str:
    """Dipten Uzaklık (%) ve ATH'ye Uzaklık (%)'ne göre basit dip/zirve etiketi."""
    dip = _safe_get(row, "Dipten Uzaklık (%)")
    ath = _safe_get(row, "ATH'ye Uzaklık (%)")
    try:
        dip = float(dip)
    except (TypeError, ValueError):
        dip = np.nan
    try:
        ath = float(ath)
    except (TypeError, ValueError):
        ath = np.nan

    if np.isnan(dip) and np.isnan(ath):
        return "BILINMIYOR"

    if not np.isnan(dip):
        if dip <= 15:
            return "COK_DIPTE"
        if dip <= 30:
            return "DIPTE"

    if not np.isnan(ath):
        if ath < 5:
            return "COK_YUKSEKTE"
        if ath <= 15:
            return "YUKSEKTE"

    return "ORTA_BANT"


def fiyat_konum_mesaj(row: pd.Series) -> str:
    kat = _safe_get(row, "FiyatKonum", "")
    dip = _safe_get(row, "Dipten Uzaklık (%)")
    ath = _safe_get(row, "ATH'ye Uzaklık (%)")

    try:
        dip = float(dip)
    except (TypeError, ValueError):
        dip = np.nan
    try:
        ath = float(ath)
    except (TypeError, ValueError):
        ath = np.nan

    if kat == "COK_DIPTE":
        return f"Dip bölgesine çok yakın (Dipten Uzaklık %{dip:.1f}). Orta/uzun vade için toparlanma alanı."
    if kat == "DIPTE":
        return f"Dip bölgesine yakın (Dipten Uzaklık %{dip:.1f}). Kademeli alım bölgesi."
    if kat == "YUKSEKTE":
        return f"Görece yüksek bölgede (ATH'ye Uzaklık %{ath:.1f}). Kâr realizasyonu/düzeltme riski artmış."
    if kat == "COK_YUKSEKTE":
        return f"ATH civarına çok yakın (ATH'ye Uzaklık %{ath:.1f}). Yeni alım için riskli bölge."
    if kat == "ORTA_BANT":
        return "Fiyat ne dipte ne zirvede; orta bantta. Trend ve momentum sinyalleri belirleyici."
    return "Fiyat konumu net değil; Dip/ATH mesafesi hesaplanamamış."


# ---------------- main ----------------

def main():
    base_dir = BASE_DIR

    multim_info = _detect_latest_multim4(base_dir)
    if not multim_info:
        print("[Hata] (teknik_grup_nextday5) multim4_YYYY-MM-DD.csv bulunamadı.")
        return

    date_str, multim_path = multim_info
    print(f"[Bilgi] (teknik_grup_nextday5) multim4 dosyası: {multim_path}")
    df_multi = pd.read_csv(multim_path)
    if "Sembol" not in df_multi.columns:
        print("[Hata] (teknik_grup_nextday5) multim4'te 'Sembol' kolonu yok.")
        return

    nextday_path = os.path.join(base_dir, NEXTDAY5_TEMPLATE.format(date=date_str))
    if not os.path.isfile(nextday_path):
        print(f"[Hata] (teknik_grup_nextday5) {os.path.basename(nextday_path)} bulunamadı.")
        return

    print(f"[Bilgi] (teknik_grup_nextday5) Nextday5 dosyası: {nextday_path}")
    df_next = pd.read_excel(nextday_path)
    if "Sembol" not in df_next.columns:
        print("[Hata] (teknik_grup_nextday5) Nextday5'te 'Sembol' yok.")
        return

    # multim4 teknik kolonları
    teknik_cols = [
        "Tarih",
        "Fiyat (Son)",
        "Sinyal",
        "Skor (Al)",
        "Skor (Sat)",
        "Elmas Formasyonu",
        "AlphaTrend",
        "MA",
        "MACD",
        "MACD_Signal",
        "MACD_Positive",
        "RSI",
        "ADX",
        "OBV_Diff",
        "BB_Low",
        "BB_Mavg",
        "BB_High",
        "BB_W",
        "EMA20_gt_EMA50_gt_EMA200",
        "Dipten Uzaklık (%)",
        "ATH'ye Uzaklık (%)",
        "ATH Breakout",
        "ATH Dip Potansiyeli",
        "Cup&Handle Formasyonu",
        "Bull Flag Formasyonu",
        "TOBO Formasyonu",
        "Gelişmiş TOBO",
        "EW_WaveType_1d",
        "EW_CurrentWave_1d",
        "EW_Phase_1d",
        "EW_LastPivotTime_1d",
        "EW_LastPivotPrice_1d",
        "EW_WaveType_4h",
        "EW_CurrentWave_4h",
        "EW_Phase_4h",
        "EW_LastPivotTime_4h",
        "EW_LastPivotPrice_4h",
        "EW_Direction_1d",
        "EW_AsOf_1d",
        "EW_Direction_4h",
        "EW_AsOf_4h",
        "SMA5_137_15m_Sinyal",
        "SMA5_137_15m_Sinyal_Numeric",
        "SMA5_137_15m_Sinyal_Filt",
        "SMA5_137_15m_Sinyal_Filt_Numeric",
        "SMA5_137_15m_CrossTime",
        "SMA5_137_15m_BarsSinceCross",
        "SMA5_137_1h_Sinyal",
        "SMA5_137_1h_Sinyal_Numeric",
        "SMA5_137_1h_Sinyal_Filt",
        "SMA5_137_1h_Sinyal_Filt_Numeric",
        "SMA5_137_1h_CrossTime",
        "SMA5_137_1h_BarsSinceCross",
        "SMA5_137_4h_Sinyal",
        "SMA5_137_4h_Sinyal_Numeric",
        "SMA5_137_4h_Sinyal_Filt",
        "SMA5_137_4h_Sinyal_Filt_Numeric",
        "SMA5_137_4h_CrossTime",
        "SMA5_137_4h_BarsSinceCross",
        "SMA5_137_1d_Sinyal",
        "SMA5_137_1d_Sinyal_Numeric",
        "SMA5_137_1d_Sinyal_Filt",
        "SMA5_137_1d_Sinyal_Filt_Numeric",
        "SMA5_137_1d_CrossTime",
        "SMA5_137_1d_BarsSinceCross",
        "CandleGoldenCross_Over",
        "Confirmed_BUY",
        "Confirmed_SELL",
        "CandleGoldenCross_Over_Confirmed_BUY",
        "Confirmed_SELL_CandleCrossUnder",
        "STOP Fiyat",
        "TP Fiyat",
        "Haftalık Değişim (%)",
        "Aylık Değişim (%)",
        "Ortalama Hacim (TL)",
        "F/K",
        "PD/DD",
        "Yorumlar",
        "PUANLAMA_V4",
        "TP Oran (%)",
        "Sinyal_Sort",
        "VolumeDeltaUp_Sort",
        "Formasyon_Sort",
        "FormasyonTazeKirilim",
        "BONUS_SCORE",
        "DipSkor",
        "ATHSkor",
        "OBVSkor",
        "TazeSkor",
        "FinalSkorEx",
    ]

    use_cols = ["Sembol"]
    for c in teknik_cols:
        if c in df_multi.columns:
            use_cols.append(c)
    df_multi_small = df_multi[use_cols].copy().drop_duplicates("Sembol", keep="last")

    df_merged = pd.merge(
        df_next,
        df_multi_small,
        on="Sembol",
        how="left",
        suffixes=("", "_m4"),
    )

    if df_merged.empty:
        print("[Uyarı] (teknik_grup_nextday5) Birleşmiş satır yok.")
        return

    # Sayısal dönüşümler
    for col in [
        "MACD", "MACD_Signal", "MACD_Positive",
        "RSI", "ADX",
        "Dipten Uzaklık (%)",
        "ATH'ye Uzaklık (%)",
        "BB_W",
        "AlphaTrend",
        "Fiyat_Guncel",
        "Sinyal",
        "PUANLAMA_V4",
        "FinalSkorEx",
    ]:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    # Skorlar
    df_merged["TrendSkor"] = df_merged.apply(hesapla_trend_skor, axis=1)
    df_merged["MomentumSkor"] = df_merged.apply(hesapla_momentum_skor, axis=1)
    df_merged["RiskSkor"] = df_merged.apply(hesapla_risk_skor, axis=1)
    df_merged["SinyalSkor"] = df_merged.apply(hesapla_sinyal_skor, axis=1)
    df_merged["FormasyonSkor"] = df_merged.apply(hesapla_formasyon_skor, axis=1)
    df_merged["TeknikSkor"] = df_merged.apply(hesapla_toplam_teknik_skor, axis=1)
    df_merged["TeknikGrup"] = df_merged.apply(teknik_grup_etk, axis=1)

    df_merged["FiyatKonum"] = df_merged.apply(fiyat_konum_kategori, axis=1)
    df_merged["FiyatKonumMesaj"] = df_merged.apply(fiyat_konum_mesaj, axis=1)

    # Kolon sırası: ilk 9 kolon sabit, sonra 4 özel kolon, sonra kalanlar
    cols = list(df_merged.columns)
    fixed_prefix = cols[:9]
    special = ["TeknikSkor", "TeknikGrup", "FiyatKonum", "FiyatKonumMesaj"]
    remaining = [c for c in cols if c not in fixed_prefix + special]
    new_order = fixed_prefix + special + remaining
    df_merged = df_merged[new_order]

    df_merged = df_merged.sort_values(
        by=["TeknikGrup", "TeknikSkor"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Aynı Excel dosyasında yeni sheet olarak yaz
    out_path = nextday_path
    try:
        import openpyxl  # noqa: F401

        with pd.ExcelWriter(
            out_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            df_merged.to_excel(writer, index=False, sheet_name=SHEET_NAME)

        print(f"[ÇIKTI] (teknik_grup_nextday5) Teknik grup sheet'i eklendi: {out_path} ({SHEET_NAME})")
    except ModuleNotFoundError:
        print("[Uyarı] (teknik_grup_nextday5) openpyxl yok, Excel güncellenemedi.")


if __name__ == "__main__":
    main()