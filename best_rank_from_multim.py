import os
import re
from datetime import datetime, date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MULTIM_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
BACKTEST5_PATTERN = r"^GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.csv$"

OUT_TEMPLATE_XLSX = "BEST_RANK_FROM_MULTIM_{date}.xlsx"
OUT_TEMPLATE_CSV = "BEST_RANK_FROM_MULTIM_{date}.csv"

TOP_K = 20  # Kaç hisse seçilecek


def _detect_latest_by_pattern(folder: str, pattern: str) -> Optional[Tuple[str, str]]:
    rx = re.compile(pattern, re.IGNORECASE)
    candidates: List[Tuple[date, str]] = []
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


def _to_num(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def compute_rank_score(row: pd.Series) -> float:
    """
    Backtest feature analizinde en iyi çıkan kolonlara göre
    son multim4 satırı için birleşik skor.
    Yüksek skor = daha cazip.
    Kullanılan ana kolonlar:
      - PUANLAMA_V4
      - Skor (Al)
      - Sinyal_Sort
      - RSI
      - MACD_Signal
      - MACD
      - FinalSkorEx (opsiyonel katkı)
    """

    skor = 0.0

    # 1) Genel puanlar
    p4 = row.get("PUANLAMA_V4")
    fex = row.get("FinalSkorEx")
    try:
        if not pd.isna(p4):
            # 0-10 gibi varsa, 0-5 arası katkı
            skor += np.clip(float(p4) / 2.0, -2.0, 5.0)
    except Exception:
        pass

    try:
        if not pd.isna(fex):
            v = float(fex)
            if v > 0:
                skor += np.clip(v / 10.0, 0.0, 4.0)
            else:
                skor += np.clip(v / 20.0, -2.0, 0.0)
    except Exception:
        pass

    # 2) MACD tarafı (momentum/trend)
    macd = row.get("MACD")
    macd_sig = row.get("MACD_Signal")
    macd_pos = row.get("MACD_Positive")

    try:
        if not pd.isna(macd_sig):
            v = float(macd_sig)
            skor += np.clip(v / 2.0, -3.0, 3.0)
    except Exception:
        pass

    try:
        if not pd.isna(macd):
            v = float(macd)
            skor += np.clip(v / 2.0, -2.0, 2.0)
    except Exception:
        pass

    if macd_pos in (1, "1", True):
        skor += 1.5

    # 3) RSI: orta bant iyidir
    rsi = row.get("RSI")
    try:
        rsi = float(rsi)
        if 40 <= rsi <= 60:
            skor += 2.0
        elif 30 <= rsi < 40 or 60 < rsi <= 75:
            skor += 1.0
        elif rsi > 80:
            skor -= 2.0
        elif rsi < 25:
            skor -= 1.0
    except Exception:
        pass

    # 4) Sinyal skorları
    skor_al = row.get("Skor (Al)")
    sinyal_sort = row.get("Sinyal_Sort")
    try:
        if not pd.isna(skor_al):
            skor += np.clip(float(skor_al) / 20.0, -2.0, 4.0)
    except Exception:
        pass

    try:
        if not pd.isna(sinyal_sort):
            skor += np.clip(float(sinyal_sort) / 20.0, -1.0, 3.0)
    except Exception:
        pass

    # 5) Trend yapısı: EMA20>EMA50>EMA200
    if row.get("EMA20_gt_EMA50_gt_EMA200") in (1, "1", True):
        skor += 2.0

    # 6) Dip / ATH konumu (aşırı uçlara hafif ceza)
    dip = row.get("Dipten Uzaklık (%)")
    ath = row.get("ATH'ye Uzaklık (%)")
    try:
        dip = float(dip)
    except Exception:
        dip = np.nan
    try:
        ath = float(ath)
    except Exception:
        ath = np.nan

    if not np.isnan(dip):
        if dip <= 30:
            skor += 1.0
        elif dip > 85:
            skor -= 1.5

    if not np.isnan(ath):
        if 8 <= ath <= 60:
            skor += 0.5
        elif ath < 3:
            skor -= 1.0
        elif ath > 95:
            skor -= 1.0

    return float(skor)


def main():
    # 1) Son backtest dosyasını bul (şimdilik sadece tarih için kullanıyoruz)
    back_info = _detect_latest_by_pattern(BASE_DIR, BACKTEST5_PATTERN)
    if not back_info:
        print("[Uyarı] GERCEK_BACKTEST5_TO_LATEST_*.csv bulunamadı, sadece multim tarihi kullanılacak.")
        back_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    else:
        back_date_str, back_path = back_info
        print(f"[Bilgi] Son backtest5 dosyası: {back_path} (tarih={back_date_str})")

    # 2) Son multim4
    multi_info = _detect_latest_by_pattern(BASE_DIR, MULTIM_PATTERN)
    if not multi_info:
        print("[Hata] multim4_YYYY-MM-DD.csv bulunamadı.")
        return
    multim_date_str, multim_path = multi_info
    print(f"[Bilgi] Son multim4 dosyası: {multim_path} (tarih={multim_date_str})")

    # Çıktı tarihini multim tarihiyle eşleyelim
    out_date = multim_date_str

    df = pd.read_csv(multim_path, low_memory=False)
    if "Sembol" not in df.columns:
        print("[Hata] multim4 dosyasında 'Sembol' kolonu yok.")
        return

    df["Sembol"] = df["Sembol"].astype(str).str.strip().str.upper()

    # İlgili numerik kolonları sayıya çevir
    numeric_cols = [
        "PUANLAMA_V4",
        "FinalSkorEx",
        "MACD",
        "MACD_Signal",
        "MACD_Positive",
        "RSI",
        "Skor (Al)",
        "Sinyal_Sort",
        "Dipten Uzaklık (%)",
        "ATH'ye Uzaklık (%)",
    ]
    _to_num(df, numeric_cols)

    # Hafif teknik filtre (çok da sert değil)
    mask = pd.Series(True, index=df.index)

    # MACD_Positive = 1 ise pozitif trend
    if "MACD_Positive" in df.columns:
        mask &= df["MACD_Positive"].isin([1, "1", True])

    # RSI 30-80 aralığı
    if "RSI" in df.columns:
        mask &= (df["RSI"] >= 30) & (df["RSI"] <= 80)

    # EMA sıralama
    if "EMA20_gt_EMA50_gt_EMA200" in df.columns:
        mask &= df["EMA20_gt_EMA50_gt_EMA200"].isin([1, "1", True])

    df_filt = df[mask].copy()
    print(f"[Bilgi] Hafif teknik filtre sonrası kalan satır: {len(df_filt)} / {len(df)}")

    if df_filt.empty:
        print("[Uyarı] Filtre sonrası hiç hisse kalmadı. Filtreleri yumuşatmak gerekebilir.")
        return

    # Rank skoru hesapla
    df_filt["Rank_Skor"] = df_filt.apply(compute_rank_score, axis=1)

    # En iyi TOP_K hisseyi seç
    df_sorted = df_filt.sort_values("Rank_Skor", ascending=False).reset_index(drop=True)
    top = df_sorted.head(TOP_K).copy()

    # Sayısal kolonları yuvarla
    num_cols_top = [c for c in top.columns if pd.api.types.is_numeric_dtype(top[c])]
    if num_cols_top:
        top[num_cols_top] = top[num_cols_top].round(2)

    # Kolon sırası: önemli olanları öne al
    base_cols = ["Sembol"]
    key_cols = [
        "Rank_Skor",
        "PUANLAMA_V4",
        "FinalSkorEx",
        "Skor (Al)",
        "Sinyal_Sort",
        "MACD",
        "MACD_Signal",
        "MACD_Positive",
        "RSI",
        "Dipten Uzaklık (%)",
        "ATH'ye Uzaklık (%)",
        "EMA20_gt_EMA50_gt_EMA200",
        "Fiyat (Son)",
    ]
    ordered: List[str] = []
    for c in base_cols + key_cols:
        if c in top.columns and c not in ordered:
            ordered.append(c)
    for c in top.columns:
        if c not in ordered:
            ordered.append(c)

    top = top[ordered]

    out_xlsx = os.path.join(BASE_DIR, OUT_TEMPLATE_XLSX.format(date=out_date))
    out_csv = os.path.join(BASE_DIR, OUT_TEMPLATE_CSV.format(date=out_date))

    # Çıktı yaz
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            top.to_excel(w, index=False, sheet_name="BEST_RANK")
        print(f"[ÇIKTI] BEST_RANK listesi (XLSX): {out_xlsx}")
    except ModuleNotFoundError:
        top.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[ÇIKTI] BEST_RANK listesi (CSV): {out_csv}")


if __name__ == "__main__":
    main()