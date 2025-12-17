import os
import re
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Çeşitli seçim çıktıları için pattern'ler
PATTERNS = [
    # Ana listeler (backtest5.py'den)
    (r"^LISTE5_2_5GUN_YUKSEK_RISK_TOP20_(\d{4}-\d{2}-\d{2})\.csv$", "LISTE5_YUKSEK_RISK_5G"),
    (r"^LISTE5_3_15GUN_DUSUK_RISK_MIN20_(\d{4}-\d{2}-\d{2})\.csv$", "LISTE5_DUSUK_RISK_15G"),
    (r"^RISKLI5_5GUN_VE_15GUN_BIRLESIK_(\d{4}-\d{2}-\d{2})\.xlsx$", "RISKLI5_BIRLESIK"),

    # Yeni Avcı Stratejileri
    (r"^DIP_AVCISI_LISTE_(\d{4}-\d{2}-\d{2})\.xlsx$", "DIP_AVCISI"),
    (r"^SIKISAN_PATLAMA_LISTESI_(\d{4}-\d{2}-\d{2})\.xlsx$", "SIKISAN_PATLAMA"),

    # Diğer Yardımcı Listeler
    (r"^BEST_AUTO_COMBO_LISTE_(\d{4}-\d{2}-\d{2})\.xlsx$", "BEST_AUTO_COMBO"),
    (r"^TEKNIK_TABANLI_SECIM_nextday5_(\d{4}-\d{2}-\d{2})\.xlsx$", "TEKNIK_TABANLI"),
    (r"^YARIN5_TEKNIK_LISTE_(\d{4}-\d{2}-\d{2})\.xlsx$", "YARIN5_TEKNIK"),
    (r"^BEST_RANK_FROM_MULTIM_(\d{4}-\d{2}-\d{2})\.xlsx$", "BEST_RANK_MULTIM"),
    (r"^PATTERN_NEXTDAY5_(\d{4}-\d{2}-\d{2})\.xlsx$", "PATTERN_NEXTDAY5"),
]

OUT_TEMPLATE = "BIRLESIK_TUM_LISTELER_{date}.xlsx"


def _detect_latest_by_pattern(folder: str, pattern: str) -> Optional[tuple[str, str]]:
    """Verilen pattern'e uyan en güncel dosyayı bulur."""
    rx = re.compile(pattern, re.IGNORECASE)
    candidates: List[tuple[date, str]] = []
    for fname in os.listdir(folder):
        m = rx.match(fname)
        if m:
            d_str = m.group(1)
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                candidates.append((d, fname))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    d, fname = candidates[-1]
    return d.strftime("%Y-%m-%d"), os.path.join(folder, fname)


def _read_any_file(path: str) -> Optional[pd.DataFrame]:
    """CSV veya Excel dosyasını okur."""
    ext = os.path.splitext(path.lower())[1]
    df = None
    try:
        if ext == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            # Excel dosyasındaki tüm sheet'leri birleştir
            xls = pd.ExcelFile(path, engine="openpyxl")
            sheet_dfs = [pd.read_excel(xls, sheet_name=sh) for sh in xls.sheet_names]
            if sheet_dfs:
                df = pd.concat(sheet_dfs, ignore_index=True)
    except Exception as e:
        print(f"[Uyarı] '{os.path.basename(path)}' okunamadı: {e}")
        return None

    if df is not None and "Sembol" in df.columns:
        df["Sembol"] = df["Sembol"].astype(str).str.strip().str.upper()
        return df
    return None


def _collect_from_file(path: str, label: str) -> pd.DataFrame:
    """Bir seçim dosyasından temel bilgileri alır."""
    df = _read_any_file(path)
    if df is None or df.empty:
        return pd.DataFrame()

    df_small = pd.DataFrame()
    df_small["Sembol"] = df["Sembol"]
    df_small["KaynakListe"] = label

    # Fiyat bilgisini al
    price_col = next((c for c in df.columns if c.lower() in ["fiyat (son)", "fiyat_guncel"]), None)
    if price_col:
        df_small["Fiyat"] = df[price_col]

    # Skorları al
    score_cols = [c for c in df.columns if "skor" in c.lower()]
    if score_cols:
        df_small["Maks_Skor"] = df[score_cols].max(axis=1)

    return df_small


def main():
    base_dir = BASE_DIR
    all_selections = []

    # 1. Tüm pattern'lere uyan en güncel dosyaları bul ve verilerini topla
    for pat, label in PATTERNS:
        file_info = _detect_latest_by_pattern(base_dir, pat)
        if file_info:
            _, path = file_info
            print(f"[Bilgi] Birleştiriliyor: {os.path.basename(path)} (Etiket: {label})")
            df_part = _collect_from_file(path, label)
            if not df_part.empty:
                all_selections.append(df_part)

    if not all_selections:
        print("[Hata] Birleştirilecek hiçbir seçim dosyası bulunamadı.")
        return

    # 2. Tüm verileri tek bir DataFrame'de birleştir
    df_all = pd.concat(all_selections, ignore_index=True)
    df_all["Sembol"] = df_all["Sembol"].str.strip().str.upper()

    # 3. Her sembol için kaynak listelerini ve sayısını birleştir
    agg_funcs = {
        "KaynakListe": lambda s: ",".join(sorted(set(s.dropna()))),
        "Fiyat": "last", # Son bulunan fiyatı al
        "Maks_Skor": "max" # En yüksek skoru al
    }
    # Var olmayan kolonlar için agg_funcs'tan çıkar
    valid_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_all.columns}

    df_merged = df_all.groupby("Sembol").agg(valid_agg_funcs).reset_index()

    df_merged["KaynakSayisi"] = df_merged["KaynakListe"].apply(lambda x: len(x.split(',')))

    # 4. En son multim4'ten diğer tüm teknik verileri ekle
    multim_info = _detect_latest_by_pattern(base_dir, r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$")
    if multim_info:
        date_str, multim_path = multim_info
        df_multi = pd.read_csv(multim_path, low_memory=False)
        df_multi["Sembol"] = df_multi["Sembol"].astype(str).str.strip().str.upper()

        # Fiyatı multim4'ten güncelle/doldur
        if "Fiyat" in df_merged.columns and "Fiyat (Son)" in df_multi.columns:
            df_multi_price = df_multi[["Sembol", "Fiyat (Son)"]].rename(columns={"Fiyat (Son)": "Fiyat_m4"})
            df_merged = pd.merge(df_merged, df_multi_price, on="Sembol", how="left")
            df_merged["Fiyat"] = df_merged["Fiyat_m4"].fillna(df_merged["Fiyat"])
            df_merged.drop(columns=["Fiyat_m4"], inplace=True)

        # Diğer kolonları ekle
        df_merged = pd.merge(df_merged, df_multi, on="Sembol", how="left", suffixes=("_birlesik", ""))

    # 5. Sırala: En çok kaynakta çıkanlar, sonra en yüksek skora sahip olanlar
    sort_cols = ["KaynakSayisi"]
    sort_ascending = [False]
    if "Maks_Skor" in df_merged.columns:
        sort_cols.append("Maks_Skor")
        sort_ascending.append(False)
    
    df_merged = df_merged.sort_values(by=sort_cols, ascending=sort_ascending)

    # 6. Çıktı dosyasını yaz
    date_str = date.today().strftime("%Y-%m-%d")
    out_path = os.path.join(base_dir, OUT_TEMPLATE.format(date=date_str))

    try:
        import openpyxl
        # Sayısal kolonları yuvarla
        for col in df_merged.select_dtypes(include=np.number).columns:
            df_merged[col] = df_merged[col].round(2)
            
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_merged.to_excel(writer, sheet_name="BIRLESIK_TUM_LISTELER", index=False)
        print(f"[ÇIKTI] Tüm listeler birleştirildi: {out_path}")
    except ImportError:
        print("[Uyarı] 'openpyxl' paketi kurulu değil. Excel çıktısı oluşturulamadı.")


if __name__ == "__main__":
    main()