import os
import re
from datetime import datetime, date
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# ===============================================
# Strateji: Dip Avcısı v5 (Deep Value & Turnaround)
# Amaç: Ciddi düşüş yaşamış, dip bölgesinde olan
#       ancak teknik olarak dönüş sinyali veren
#       hisseleri bulmak.
# ===============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIM4_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
OUTPUT_FILENAME_TEMPLATE = "DIP_AVCISI_LISTE_{date}.xlsx"


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
    print("[ÇALIŞTIRILIYOR] dipavcisi5.py")

    # 1. En güncel multim4 dosyasını bul ve oku
    multim_info = _detect_latest_multim4(BASE_DIR)
    if not multim_info:
        print("[Hata] dipavcisi5: multim4_YYYY-MM-DD.csv dosyası bulunamadı.")
        return

    date_str, multim_path = multim_info
    print(f"[Bilgi] dipavcisi5: Kullanılan multim4 dosyası: {multim_path}")
    df = pd.read_csv(multim_path, low_memory=False)

    # 2. Gerekli kolonları sayısal yap
    numeric_cols = [
        "Dipten Uzaklık (%)", "ATH'ye Uzaklık (%)", "RSI", "MACD_Positive",
        "VolumeDeltaUp_Sort", "Sinyal", "Haftalık Değişim (%)",
        "Aylık Değişim (%)", "PUANLAMA_V4", "FinalSkorEx", "Fiyat (Son)"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[Uyarı] dipavcisi5: Beklenen kolon '{col}' bulunamadı.")
            df[col] = np.nan

    # 3. Dip Avcısı Filtreleme Aşamaları

    # Aşama 1: Geniş Dip Havuzu
    mask_dip = (
        (df["Dipten Uzaklık (%)"] <= 20) &
        (df["ATH'ye Uzaklık (%)"] >= 40) &
        (df["Fiyat (Son)"] > 1.0)
    )
    df_filt = df[mask_dip].copy()
    print(f"[Filtre 1] Dip Havuzu: {len(df_filt)} hisse bulundu.")

    if df_filt.empty:
        print("[Bilgi] dipavcisi5: Kriterlere uygun hisse bulunamadı.")
        return

    # Aşama 2: Dönüş Emaresi Filtresi
    mask_donus = (
        (df_filt["RSI"].between(35, 60)) &
        (df_filt["MACD_Positive"] == 1) &
        (df_filt["Haftalık Değişim (%)"] > -5) &
        (df_filt["Aylık Değişim (%)"] < 30)
    )
    df_filt = df_filt[mask_donus].copy()
    print(f"[Filtre 2] Dönüş Emaresi: {len(df_filt)} hisse kaldı.")

    if df_filt.empty:
        print("[Bilgi] dipavcisi5: Kriterlere uygun hisse bulunamadı.")
        return

    # Aşama 3: Güç ve Kalite Puanlaması
    df_filt["DipAvcisi_Skor"] = (
        df_filt["VolumeDeltaUp_Sort"].fillna(0) * 1.5 +
        df_filt["Sinyal"].fillna(0) * 0.01 +
        df_filt["PUANLAMA_V4"].fillna(0) * 0.5 +
        df_filt["FinalSkorEx"].fillna(0) * 0.2 -
        df_filt["Dipten Uzaklık (%)"].fillna(20) * 0.1
    )

    df_final = df_filt.sort_values(by="DipAvcisi_Skor", ascending=False)
    top_5_hisseler = df_final.head(5)

    print("\n========== DİP AVCISI v5 - TOP 5 HİSSE ==========")
    report_cols = [
        "Sembol", "Fiyat (Son)", "DipAvcisi_Skor", "Dipten Uzaklık (%)",
        "ATH'ye Uzaklık (%)", "RSI", "PUANLAMA_V4", "VolumeDeltaUp_Sort"
    ]
    report_cols_exist = [c for c in report_cols if c in top_5_hisseler.columns]
    
    if not top_5_hisseler.empty:
        print(top_5_hisseler[report_cols_exist].to_string(index=False))

    # 4. Excel'e Yaz
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME_TEMPLATE.format(date=date_str))
    try:
        import openpyxl
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            top_5_hisseler.to_excel(writer, sheet_name="DIP_AVCILARI_TOP5", index=False)
            df_final.to_excel(writer, sheet_name="TUM_ADAYLAR", index=False)
        print(f"\n[ÇIKTI] Dip Avcısı listesi oluşturuldu: {output_path}")
    except ImportError:
        print("\n[Uyarı] 'openpyxl' paketi kurulu değil. Excel çıktısı oluşturulamadı.")

if __name__ == "__main__":
    main()