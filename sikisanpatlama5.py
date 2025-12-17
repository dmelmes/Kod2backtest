import os
import re
from datetime import datetime, date
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# =================================================================
# Strateji: Sıkışan Patlama Avcısı v5 (Coiled Spring / Breakout Hunter)
# Amaç:   Teknik olarak sıkışmış (Bollinger Bandı daralmış),
#         nötr momentumda olan ve hacimli bir alım sinyaliyle
#         patlamaya hazır hisseleri tespit etmek.
# =================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIM4_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
OUTPUT_FILENAME_TEMPLATE = "SIKISAN_PATLAMA_LISTESI_{date}.xlsx"


def _detect_latest_multim4(folder: str) -> Optional[Tuple[str, str]]:
    """Klasördeki en güncel multim4 dosyasını bulur."""
    rx = re.compile(MULTIM4_PATTERN, re.IGNORECASE)
    candidates: List[Tuple[date, str]] = []
    for fname in os.listdir(folder):
        m = rx.match(fname)
        if m:
            d_str = m.group(1)
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                candidates.append((d, os.path.join(folder, fname)))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    latest_date, latest_path = candidates[-1]
    return latest_date.strftime("%Y-%m-%d"), latest_path


def main():
    print("[ÇALIŞTIRILIYOR] sikisanpatlama5.py")

    # 1. En güncel multim4 dosyasını bul ve oku
    multim_info = _detect_latest_multim4(BASE_DIR)
    if not multim_info:
        print("[Hata] sikisanpatlama5: multim4_YYYY-MM-DD.csv dosyası bulunamadı.")
        return

    date_str, multim_path = multim_info
    print(f"[Bilgi] sikisanpatlama5: Kullanılan multim4 dosyası: {multim_path}")
    df = pd.read_csv(multim_path, low_memory=False)

    # 2. Gerekli kolonları sayısal yap
    numeric_cols = [
        "Fiyat (Son)", "PUANLAMA_V4", "FinalSkorEx", "RSI", "BB_W",
        "Sinyal", "Confirmed_BUY", "VolumeDeltaUp_Sort", "OBVSkor"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[Uyarı] sikisanpatlama5: Beklenen kolon '{col}' bulunamadı.")
            df[col] = np.nan

    # 3. Sıkışan Patlama Avcısı Filtreleme Aşamaları

    # Aşama 1: Genel Kalite Filtresi
    mask_kalite = (
        (df["PUANLAMA_V4"] >= 2.0) &
        (df["FinalSkorEx"] >= 0) &
        (df["Fiyat (Son)"] > 1.0)
    )
    df_filt = df[mask_kalite].copy()
    print(f"[Filtre 1] Genel Kalite: {len(df_filt)} hisse kaldı.")

    # Aşama 2: Sıkışma (Squeeze) Filtresi
    mask_sikisma = (
        (df_filt["BB_W"] <= 20) &
        (df_filt["RSI"].between(40, 65))
    )
    df_filt = df_filt[mask_sikisma].copy()
    print(f"[Filtre 2] Sıkışma (Squeeze): {len(df_filt)} hisse kaldı.")

    # Aşama 3: Patlama Tetiği Filtresi
    mask_tetik = (
        (df_filt["Sinyal"] == 100) | (df_filt["Confirmed_BUY"] == 1)
    ) & (
        df_filt["VolumeDeltaUp_Sort"] > 0
    )
    df_filt = df_filt[mask_tetik].copy()
    print(f"[Filtre 3] Patlama Tetiği: {len(df_filt)} hisse kaldı.")

    if df_filt.empty:
        print("[Bilgi] sikisanpatlama5: Kriterlere uygun hisse bulunamadı.")
        return

    # Aşama 4: Patlama Potansiyel Puanlaması
    df_filt["Patlama_Skoru"] = (
        df_filt["VolumeDeltaUp_Sort"].fillna(0) * 2.0 +
        df_filt["OBVSkor"].fillna(0) * 1.5 +
        df_filt["PUANLAMA_V4"].fillna(0) * 1.0 +
        (20 - df_filt["BB_W"].fillna(20)) * 0.5
    )

    df_final = df_filt.sort_values(by="Patlama_Skoru", ascending=False)
    top_5_hisseler = df_final.head(5)

    print("\n========== SIKIŞAN PATLAMA v5 - TOP 5 HİSSE ==========")
    report_cols = [
        "Sembol", "Fiyat (Son)", "Patlama_Skoru", "BB_W", "RSI",
        "VolumeDeltaUp_Sort", "OBVSkor", "Sinyal", "PUANLAMA_V4"
    ]
    report_cols_exist = [c for c in report_cols if c in top_5_hisseler.columns]
    
    if not top_5_hisseler.empty:
        print(top_5_hisseler[report_cols_exist].to_string(index=False))

    # 5. Excel'e Yaz
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME_TEMPLATE.format(date=date_str))
    try:
        import openpyxl
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            top_5_hisseler.to_excel(writer, sheet_name="SIKISAN_PATLAMA_TOP5", index=False)
            df_final.to_excel(writer, sheet_name="TUM_ADAYLAR", index=False)
        print(f"\n[ÇIKTI] Sıkışan Patlama listesi oluşturuldu: {output_path}")
    except ImportError:
        print("\n[Uyarı] 'openpyxl' paketi kurulu değil. Excel çıktısı oluşturulamadı.")

if __name__ == "__main__":
    main()