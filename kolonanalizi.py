import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MULTIM_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
BACKTEST_PATTERN = r"^GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.(csv|xlsx)$"

# Hedef: 5 güne kadar >= %10
THRESH_5D = 10.0

# Her gün kaç hisse seçeceğiz? (farklı N'ler denenir)
TOP_N_CANDIDATES = [10, 20, 30, 50]

# Kombinasyon analizi için beyaz listeye girebilecek kolon adlarında arayacağımız parçalar
COMBO_WHITELIST_CONTAINS = [
    # skorlar
    "PUANLAMA_V4",
    "FinalSkorEx",
    "FinalSkorPlus",
    "Skor (Al)",
    "Skor (Sat)",
    "Sinyal_Sort",
    "Sinyal",

    # teknik indikatörler
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Positive",
    "ADX",

    # ortalamalar / bantlar / trend
    "MA",
    "BB_Low",
    "BB_Mavg",
    "BB_High",
    "BB_W",
    "EMA20_gt_EMA50_gt_EMA200",

    # seviye / uzaklık
    "Dipten Uzaklık (%)",
    "DipSkor",
    "ATH'ye Uzaklık (%)",
    "ATHSkor",
    "ATH Dip Potansiyeli",

    # hacim/OBV
    "OBV_Diff",
    "OBVSkor",
    "VolumeDeltaUp_Sort",

    # formasyonlar / bonus skorlar
    "Cup&Handle Formasyonu",
    "Bull Flag Formasyonu",
    "TOBO Formasyonu",
    "Elmas Formasyonu",
    "Gelişmiş TOBO",
    "BONUS_SCORE",
    "Formasyon_Sort",
    "FormasyonTazeKirilim",
    "TazeSkor",

    # sinyal flagleri
    "Confirmed_BUY",
    "Confirmed_SELL",
    "Confirmed_SELL_CandleCrossUnder",
    "CandleGoldenCross_Over",
    "CandleGoldenCross_Over_Confirmed_BUY",

    # geri kalan, anlamlı olabilecekler
    "Ortalama Hacim (TL)",
    "F/K",
    "PD/DD",
    "ATH Breakout",
]

# FDTT / kurum / AKD / turnover vs ile ilgili kolonları dışlamak için aranacak parçalar
HARD_EXCLUDE_CONTAINS = [
    "__FDTT",
    "__AKD",
    "__FF_",
    "__TURNOVER",
    "__KURUM",
    "KURUM_AKD_SKOR",
]

# Günlük / haftalık / aylık değişim kolonlarını dışla
CHANGE_EXACT = {
    "Haftalık Değişim (%)",
    "Aylık Değişim (%)",
    "Son Gün %",
    "Son Gün %",
}


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1254", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _read_excel_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except ImportError:
        raise RuntimeError("openpyxl yüklü değil. 'pip install openpyxl' ile kur.")


def _get_all_multim_files(base_dir: str) -> List[Tuple[datetime, str]]:
    rx = re.compile(MULTIM_PATTERN, re.IGNORECASE)
    out: List[Tuple[datetime, str]] = []
    for fname in os.listdir(base_dir):
        m = rx.match(fname)
        if not m:
            continue
        d_str = m.group(1)
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        out.append((d, os.path.join(base_dir, fname)))
    out.sort(key=lambda x: x[0])
    return out


def _get_latest_backtest_file(base_dir: str) -> Optional[str]:
    rx = re.compile(BACKTEST_PATTERN, re.IGNORECASE)
    candidates: List[Tuple[datetime, str]] = []
    for fname in os.listdir(base_dir):
        m = rx.match(fname)
        if not m:
            continue
        d_str = m.group(1)
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        candidates.append((d, os.path.join(base_dir, fname)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_backtest(base_dir: str) -> pd.DataFrame:
    """
    GERCEK_BACKTEST5_TO_LATEST_* dosyasını okur ve
    Sembol + Analiz_Tarihi_str bazında 5 günlük max getiri label'ı hazırlar.
    (Label_5g = Max_Getiri_% veya Getiri_1Hafta_Sonu_%)
    """
    path = _get_latest_backtest_file(base_dir)
    if not path:
        raise FileNotFoundError("GERCEK_BACKTEST5_TO_LATEST_* dosyası bulunamadı.")

    print(f"[Bilgi] Backtest dosyası: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = _read_csv_any(path)
    else:
        df = _read_excel_any(path)

    if "Sembol" not in df.columns:
        raise ValueError("Backtest dosyasında 'Sembol' kolonu yok.")

    df["Sembol"] = df["Sembol"].astype(str).str.strip().str.upper()

    if "Analiz_Tarihi_str" not in df.columns:
        raise ValueError("Backtest dosyasında 'Analiz_Tarihi_str' kolonu yok.")
    df["Analiz_Tarihi_str"] = df["Analiz_Tarihi_str"].astype(str).str.slice(0, 10)

    col_5d = None
    for c in ["Max_Getiri_%", "Max_Getiri_5Gun_%", "Getiri_1Hafta_Sonu_%"]:
        if c in df.columns:
            col_5d = c
            break

    if col_5d is None:
        raise ValueError("Backtest dosyasında 5g getirilerini temsil eden kolon bulunamadı.")

    df["Label_5g"] = pd.to_numeric(df[col_5d], errors="coerce")

    out = df[["Sembol", "Analiz_Tarihi_str", "Label_5g"]].copy()
    return out


def _merge_multim_with_backtest(multim_files: List[Tuple[datetime, str]], df_back: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm multim4_*.csv dosyalarını okuyup,
    Sembol+Analiz_Tarihi_str ile backtest etiketini (Label_5g) birleştirir.
    """
    records = []
    for d, path in multim_files:
        print(f"[Bilgi] Multim dosyası işleniyor: {path}")
        df = _read_csv_any(path)
        if df is None or df.empty or "Sembol" not in df.columns:
            print(f"[Uyarı] {path} uygun değil, atlandı.")
            continue
        df["Sembol"] = df["Sembol"].astype(str).str.strip().str.upper()

        if "Analiz_Tarihi_str" in df.columns:
            df["Analiz_Tarihi_str"] = df["Analiz_Tarihi_str"].astype(str).str.slice(0, 10)
        else:
            df["Analiz_Tarihi_str"] = d.strftime("%Y-%m-%d")

        df_merged = df.merge(
            df_back,
            on=["Sembol", "Analiz_Tarihi_str"],
            how="left",
            suffixes=("", "_bt"),
        )

        records.append(df_merged)

    if not records:
        raise RuntimeError("Hiç multim + backtest birleşimi oluşturulamadı.")

    df_all = pd.concat(records, ignore_index=True)

    # 5 güne kadar >= %10
    df_all["Hit_5g_10p"] = (df_all["Label_5g"] >= THRESH_5D).astype(float)

    return df_all


def _get_candidate_features(df_all: pd.DataFrame) -> List[str]:
    """
    Multim içindeki kolonlardan:
    - Sembol, tarih, label ve açıkça yardımcı kolonları hariç,
    - FDTT / AKD / KURUM / TURNOVER / değişim kolonlarını hariç,
    - Sayısal kolonları feature adayı olarak seçer.
    """
    exclude_exact = {
        "Sembol",
        "Analiz_Tarihi_str",
        "Label_5g",
        "Hit_5g_10p",
        "Kaynak",
        "KaynakListe",
        "KaynakSayisi",
    } | CHANGE_EXACT

    exclude_contains = [
        "Tarih",
        "Date",
        "Time",
        "Yorum",
        "Comment",
        "Label_",
    ]

    features = []
    for col in df_all.columns:
        if col in exclude_exact:
            continue
        if any(substr in col for substr in exclude_contains):
            continue
        if any(substr in col for substr in HARD_EXCLUDE_CONTAINS):
            continue
        if pd.api.types.is_numeric_dtype(df_all[col]):
            features.append(col)

    print(f"[Bilgi] Analiz edilecek feature sayısı (filtrelenmiş): {len(features)}")
    return features


def _evaluate_single_feature(df_all: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Belirli bir feature için:
    - ascending / descending
    - farklı TOP_N_CANDIDATES
    kombinasyonlarını dener.
    Her kombinasyon için:
      - mean_5g
      - hit_5g_10p
    hesaplar.
    """
    results: List[Dict[str, Any]] = []

    if df_all[feature].isna().all():
        return pd.DataFrame()

    grouped = df_all.groupby("Analiz_Tarihi_str")

    for ascending in [False, True]:  # False: büyük->küçük, True: küçük->büyük
        direction = "asc" if ascending else "desc"

        for top_n in TOP_N_CANDIDATES:
            selected_rows = []
            for _, g in grouped:
                g_valid = g.dropna(subset=[feature])
                if g_valid.empty:
                    continue
                g_sorted = g_valid.sort_values(feature, ascending=ascending)
                selected_rows.append(g_sorted.head(top_n))

            if not selected_rows:
                continue

            df_sel = pd.concat(selected_rows, ignore_index=True)

            if df_sel["Label_5g"].notna().sum() < 10:
                continue

            mean_5g = df_sel["Label_5g"].mean()
            hit_5g = df_sel["Hit_5g_10p"].mean()

            res = dict(
                feature=feature,
                direction=direction,
                top_n=top_n,
                count=len(df_sel),
                mean_5g=round(float(mean_5g) if pd.notna(mean_5g) else np.nan, 4),
                hit_5g_10p=round(float(hit_5g) if pd.notna(hit_5g) else np.nan, 4),
            )
            results.append(res)

    return pd.DataFrame(results)


def _filter_for_combos(features: List[str]) -> List[str]:
    """
    Kombinasyon analizi için, sadece "anlamlı" olan feature'ları seç.
    COMBO_WHITELIST_CONTAINS listesinde geçen parçaları kullanan kolonlar.
    """
    selected = []
    for col in features:
        if any(key in col for key in COMBO_WHITELIST_CONTAINS):
            selected.append(col)

    print(f"[Bilgi] 2'li kombinasyon için seçilen feature sayısı: {len(selected)}")
    return selected


def _evaluate_feature_combo(df_all: pd.DataFrame, f1: str, f2: str) -> pd.DataFrame:
    """
    İki feature için (f1, f2):
    - her ikisi için asc/desc yönleri,
    - TOP_N_CANDIDATES üst sınırları
    üzerinde tarama yapar.
    """
    results: List[Dict[str, Any]] = []

    if df_all[f1].isna().all() or df_all[f2].isna().all():
        return pd.DataFrame()

    grouped = df_all.groupby("Analiz_Tarihi_str")

    for asc1 in [False, True]:
        for asc2 in [False, True]:
            dir1 = "asc" if asc1 else "desc"
            dir2 = "asc" if asc2 else "desc"

            for top_n in TOP_N_CANDIDATES:
                selected_rows = []
                for _, g in grouped:
                    g_valid = g.dropna(subset=[f1, f2])
                    if g_valid.empty:
                        continue
                    g_sorted = g_valid.sort_values(
                        by=[f1, f2],
                        ascending=[asc1, asc2],
                    )
                    selected_rows.append(g_sorted.head(top_n))

                if not selected_rows:
                    continue

                df_sel = pd.concat(selected_rows, ignore_index=True)
                if df_sel["Label_5g"].notna().sum() < 10:
                    continue

                mean_5g = df_sel["Label_5g"].mean()
                hit_5g = df_sel["Hit_5g_10p"].mean()

                res = dict(
                    feature1=f1,
                    feature2=f2,
                    dir1=dir1,
                    dir2=dir2,
                    top_n=top_n,
                    count=len(df_sel),
                    mean_5g=round(float(mean_5g) if pd.notna(mean_5g) else np.nan, 4),
                    hit_5g_10p=round(float(hit_5g) if pd.notna(hit_5g) else np.nan, 4),
                )
                results.append(res)

    return pd.DataFrame(results)


def main():
    base_dir = BASE_DIR

    multim_files = _get_all_multim_files(base_dir)
    if not multim_files:
        print("[Hata] multim4_*.csv dosyası bulunamadı.")
        return

    df_back = _load_backtest(base_dir)
    df_all = _merge_multim_with_backtest(multim_files, df_back)
    print(f"[Bilgi] Toplam birleşik satır sayısı: {len(df_all)}")

    # 1) Feature adayları
    features = _get_candidate_features(df_all)

    # 2) Tek kolon analizi
    all_single_results: List[pd.DataFrame] = []
    for i, feat in enumerate(features, start=1):
        print(f"[Analiz - Tek kolon] {i}/{len(features)} - feature: {feat}")
        df_feat = _evaluate_single_feature(df_all, feat)
        if not df_feat.empty:
            all_single_results.append(df_feat)

    if not all_single_results:
        print("[Hata] Hiç tek kolon sonucu üretilmedi.")
        return

    df_single = pd.concat(all_single_results, ignore_index=True)
    df_single_sorted = df_single.sort_values(
        ["hit_5g_10p", "mean_5g"],
        ascending=[False, False],
    )
    out_single = os.path.join(base_dir, "BEST_SINGLE_FEATURES_5G_10P.csv")
    df_single_sorted.to_csv(out_single, index=False, encoding="utf-8-sig")
    print(f"[ÇIKTI] 5 gün >= %10 için en iyi TEK feature'lar: {out_single}")

    # 3) 2'li kombinasyon analizi
    combo_features = _filter_for_combos(features)

    all_combo_results: List[pd.DataFrame] = []
    n_feats = len(combo_features)
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            f1 = combo_features[i]
            f2 = combo_features[j]
            print(f"[Analiz - 2'li combo] ({i+1}/{n_feats}) {f1} + {f2}")
            df_combo = _evaluate_feature_combo(df_all, f1, f2)
            if not df_combo.empty:
                all_combo_results.append(df_combo)

    if all_combo_results:
        df_combo_all = pd.concat(all_combo_results, ignore_index=True)
        df_combo_sorted = df_combo_all.sort_values(
            ["hit_5g_10p", "mean_5g"],
            ascending=[False, False],
        )
        out_combo = os.path.join(base_dir, "BEST_COMBOS_5G_10P.csv")
        df_combo_sorted.to_csv(out_combo, index=False, encoding="utf-8-sig")
        print(f"[ÇIKTI] 5 gün >= %10 için en iyi 2'li KOMBO'lar: {out_combo}")
    else:
        print("[Uyarı] Hiç 2'li kombinasyon sonucu üretilemedi.")


if __name__ == "__main__":
    main()