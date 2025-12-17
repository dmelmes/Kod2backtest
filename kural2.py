import os
import re
from datetime import datetime, date
from typing import List, Tuple, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dosya adları regex (CSV ve XLSX)
CSV_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.(csv|CSV)$"
XLSX_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.(xlsx|XLSX)$"


def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    CSV'yi sağlam şekilde oku: olası ayraç ve tırnak sorunları için denemeler.
    Varsayılan olarak python engine + on_bad_lines='skip' kullanır.
    """
    seps = [",", ";", "\t"]
    quotes = ['"', "'"]
    encodings = ("utf-8-sig", "utf-8", "latin1", "cp1254", "cp1252")
    last_err: Optional[Exception] = None
    for enc in encodings:
        for sep in seps:
            for quote in quotes:
                try:
                    return pd.read_csv(
                        path,
                        encoding=enc,
                        sep=sep,
                        quotechar=quote,
                        engine="python",
                        on_bad_lines="skip",
                    )
                except Exception as e:
                    last_err = e
                    continue
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
            continue
    # Son çare
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _read_xlsx(path: str) -> pd.DataFrame:
    import openpyxl  # noqa: F401
    return pd.read_excel(path, engine="openpyxl")


def _find_all(base_dir: str, pattern: str) -> List[Tuple[datetime, str]]:
    rx = re.compile(pattern, re.IGNORECASE)
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
    return candidates


def _pick_today_first_else_latest() -> Tuple[str, str, str]:
    """
    Bugünün tarihli dosyayı önceliklendirir ve CSV'ye öncelik verir:
    1) Önce bugünün CSV (varsa),
    2) Yoksa bugünün XLSX (varsa),
    3) Yoksa en son CSV,
    4) Yoksa en son XLSX.
    Dönüş: (date_str, path, ext) ext=".csv" veya ".xlsx"
    """
    today_str = date.today().strftime("%Y-%m-%d")

    csv_files = _find_all(BASE_DIR, CSV_PATTERN)
    xlsx_files = _find_all(BASE_DIR, XLSX_PATTERN)

    # Bugünün adayları
    today_csv = [(d, p) for (d, p) in csv_files if d.strftime("%Y-%m-%d") == today_str]
    today_xlsx = [(d, p) for (d, p) in xlsx_files if d.strftime("%Y-%m-%d") == today_str]

    if today_csv:
        d, p = today_csv[0]
        return d.strftime("%Y-%m-%d"), p, ".csv"

    if today_xlsx:
        d, p = today_xlsx[0]
        return d.strftime("%Y-%m-%d"), p, ".xlsx"

    # Bugün yoksa en son tarihli CSV, yoksa en son XLSX
    if csv_files:
        csv_files.sort(key=lambda x: x[0])
        d, p = csv_files[-1]
        return d.strftime("%Y-%m-%d"), p, ".csv"

    if xlsx_files:
        xlsx_files.sort(key=lambda x: x[0])
        d, p = xlsx_files[-1]
        return d.strftime("%Y-%m-%d"), p, ".xlsx"

    raise FileNotFoundError("multim4_YYYY-MM-DD.(csv|xlsx) dosyası bulunamadı.")


def _load_multim_today_or_latest() -> Tuple[pd.DataFrame, str]:
    """
    Bugünün dosyasını önceliklendirip oku; yoksa en son dosyayı oku.
    CSV varsa CSV, yoksa XLSX. (CSV tercihli)
    """
    date_str, path, ext = _pick_today_first_else_latest()

    if ext.lower() == ".csv":
        print(f"[Bilgi] Seçilen multim4 CSV: {path}")
        df = _read_csv_robust(path)
    else:
        print(f"[Bilgi] Seçilen multim4 XLSX: {path}")
        df = _read_xlsx(path)

    if df is None or df.empty:
        raise RuntimeError("multim4 dosyası boş veya okunamadı.")

    # Kolon isimlerini normalize et
    df.columns = [str(c).strip() for c in df.columns]

    if "Sembol" not in df.columns:
        raise ValueError(f"multim4 dosyasında 'Sembol' kolonu yok. Kolonlar: {list(df.columns)}")

    df["Sembol"] = df["Sembol"].astype(str).str.strip().str.upper()
    return df, date_str


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"multim dosyasında eksik kolon(lar): {missing}")


# -------------------------
# Kolon çözümleyiciler (isim farklarına tolerans)
# -------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _resolve_ema5_137_signal(df: pd.DataFrame) -> Optional[str]:
    """
    'EMA5_137 Sinyal_Numeric' kolonunu farklı yazımlardan bulmaya çalışır.
    """
    candidates_priority = [
        "EMA5_137 Sinyal_Numeric",
        "EMA5_137_Sinyal_Numeric",
        "EMA5_137 Sinyal Numeric",
        "EMA5_137-Sinyal-Numeric",
    ]
    for c in candidates_priority:
        if c in df.columns:
            return c
    target_norm = _norm("EMA5_137 Sinyal_Numeric")
    norm_map = {_norm(col): col for col in df.columns}
    if target_norm in norm_map:
        return norm_map[target_norm]
    for col in df.columns:
        n = _norm(col)
        if all(tok in n for tok in ["ema5", "137", "sinyal", "numeric"]):
            return col
    for col in df.columns:
        n = _norm(col)
        if "ema5137" in n and "sinyal" in n and ("numeric" in n or "num" in n):
            return col
    return None


# -------------------------
# KURALLAR (FinalSkorPlus opsiyonel sıralama kriteri eklendi)
# -------------------------

def _select_kural1(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    KURAL 1:
      - PUANLAMA_V4 desc
      - (ops) FinalSkorPlus desc
      - FinalSkorEx asc
    """
    base_cols = ["Sembol", "PUANLAMA_V4", "FinalSkorEx"]
    _ensure_columns(df, base_cols)

    sort_by = ["PUANLAMA_V4"]
    ascending = [False]

    if "FinalSkorPlus" in df.columns:
        sort_by.append("FinalSkorPlus")
        ascending.append(False)

    sort_by.append("FinalSkorEx")
    ascending.append(True)

    df_sorted = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    df_sel = df_sorted.head(top_n).copy()
    df_sel["Kural"] = "KURAL1_Puan_FinalEx"
    df_sel["Kural_Adi"] = "PUANLAMA_V4 desc, FinalSkorPlus(desc, ops), FinalSkorEx asc"
    return df_sel


def _select_kural2(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    KURAL 2:
      - EMA5_137 Sinyal_Numeric desc (isim farklarına tolerans)
      - (ops) FinalSkorPlus desc
      - PUANLAMA_V4 desc
    """
    ema_col = _resolve_ema5_137_signal(df)
    if ema_col is None:
        print("[Uyarı] EMA5_137 Sinyal_Numeric kolonu bulunamadı; KURAL2_EMA_Puan atlandı.")
        return pd.DataFrame()

    _ensure_columns(df, ["Sembol", "PUANLAMA_V4"])

    sort_by = [ema_col]
    ascending = [False]

    if "FinalSkorPlus" in df.columns:
        sort_by.append("FinalSkorPlus")
        ascending.append(False)

    sort_by.append("PUANLAMA_V4")
    ascending.append(False)

    df_sorted = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    df_sel = df_sorted.head(top_n).copy()
    df_sel["Kural"] = "KURAL2_EMA_Puan"
    df_sel["Kural_Adi"] = f"{ema_col} desc, FinalSkorPlus(desc, ops), PUANLAMA_V4 desc"
    return df_sel


def _select_kural3(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    KURAL 3:
      - Dipten Uzaklık (%) desc
      - EMA5_137 Sinyal_Numeric desc (isim farklarına tolerans)
      - (ops) FinalSkorPlus desc
    """
    ema_col = _resolve_ema5_137_signal(df)
    if ema_col is None:
        print("[Uyarı] EMA5_137 Sinyal_Numeric kolonu bulunamadı; KURAL3_Dip_EMA atlandı.")
        return pd.DataFrame()

    base_cols = ["Sembol", "Dipten Uzaklık (%)"]
    _ensure_columns(df, base_cols)

    sort_by = ["Dipten Uzaklık (%)", ema_col]
    ascending = [False, False]

    if "FinalSkorPlus" in df.columns:
        sort_by.append("FinalSkorPlus")
        ascending.append(False)

    df_sorted = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    df_sel = df_sorted.head(top_n).copy()
    df_sel["Kural"] = "KURAL3_Dip_EMA"
    df_sel["Kural_Adi"] = f"Dipten Uzaklık desc, {ema_col} desc, FinalSkorPlus(desc, ops)"
    return df_sel


def _select_kural4(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    KURAL 4:
      - PUANLAMA_V4 desc
      - (ops) FinalSkorPlus desc
      - VolumeDeltaUp_Sort desc
    """
    base_cols = ["Sembol", "PUANLAMA_V4", "VolumeDeltaUp_Sort"]
    _ensure_columns(df, base_cols)

    sort_by = ["PUANLAMA_V4"]
    ascending = [False]

    if "FinalSkorPlus" in df.columns:
        sort_by.append("FinalSkorPlus")
        ascending.append(False)

    sort_by.append("VolumeDeltaUp_Sort")
    ascending.append(False)

    df_sorted = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    df_sel = df_sorted.head(top_n).copy()
    df_sel["Kural"] = "KURAL4_Puan_Vol"
    df_sel["Kural_Adi"] = "PUANLAMA_V4 desc, FinalSkorPlus(desc, ops), VolumeDeltaUp_Sort desc"
    return df_sel


# -------------------------
# YENİ REITER: FinalSkorPlus odaklı genel sıralama
# -------------------------

def _select_kural5_finalplus_combo(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    KURAL5_FinalPlus_Combo:
      1) FinalSkorPlus desc
      2) PUANLAMA_V4 desc
      3) EMA5_137 Sinyal_Numeric desc (ops: isim farklarına tolerans)
      4) VolumeDeltaUp_Sort desc
      5) RSI desc
      6) ATH'ye Uzaklık (%) asc
      7) ATHSkor desc
      8) FinalSkorEx asc
    """
    if "FinalSkorPlus" not in df.columns:
        print("[Uyarı] FinalSkorPlus kolonu yok; KURAL5_FinalPlus_Combo atlandı.")
        return pd.DataFrame()

    _ensure_columns(df, ["Sembol", "FinalSkorPlus"])

    ema_col = _resolve_ema5_137_signal(df)

    sort_chain = [
        ("FinalSkorPlus", False),
        ("PUANLAMA_V4", False),
        (ema_col, False) if ema_col else (None, False),
        ("VolumeDeltaUp_Sort", False),
        ("RSI", False),
        ("ATH'ye Uzaklık (%)", True),
        ("ATHSkor", False),
        ("FinalSkorEx", True),
    ]

    real_sort = []
    real_asc = []
    for col, asc in sort_chain:
        if col and col in df.columns:
            real_sort.append(col)
            real_asc.append(asc)

    df_sorted = df.sort_values(by=real_sort, ascending=real_asc).reset_index(drop=True)
    df_sel = df_sorted.head(top_n).copy()
    df_sel["Kural"] = "KURAL5_FinalPlus_Combo"
    df_sel["Kural_Adi"] = (
        f"FinalPlus(desc) > PUANLAMA_V4(desc) > {ema_col or 'EMA5_137?(yok)'}(desc) > "
        "VolDelta(desc) > RSI(desc) > ATH uzaklık(asc) > ATHSkor(desc) > FinalSkorEx(asc)"
    )
    return df_sel


# -------------------------
# GERÇEK DİP KURALI (FinalSkorPlus opsiyonel)
# -------------------------

def _select_kural_dip_strict(df_multi: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    base_needed = [
        "Dipten Uzaklık (%)",
        "ATH'ye Uzaklık (%)",
        "VolumeDeltaUp_Sort",
        "PUANLAMA_V4",
        "FinalSkorEx",
    ]
    _ensure_columns(df_multi, ["Sembol"] + base_needed)

    df_loc = df_multi.copy()

    optional_cols = [
        "Aylık Değişim (%)",
        "Son_5g_%",
        "Haftalık Değişim (%)",
        "FinalSkorPlus",
    ]

    # Dip filtresi
    dip_max = 10.0
    ath_min = 30.0
    aylik_max = 20.0
    son5_min = 0.0
    son5_max = 15.0

    for c in base_needed + optional_cols:
        if c in df_loc.columns:
            df_loc[c] = pd.to_numeric(df_loc[c], errors="coerce")

    mask = pd.Series(True, index=df_loc.index)
    mask &= df_loc["Dipten Uzaklık (%)"] <= dip_max
    mask &= df_loc["ATH'ye Uzaklık (%)"] >= ath_min

    if "Aylık Değişim (%)" in df_loc.columns:
        mask &= (df_loc["Aylık Değişim (%)"] <= aylik_max)

    son5_col = None
    if "Son_5g_%" in df_loc.columns:
        son5_col = "Son_5g_%"
    elif "Haftalık Değişim (%)" in df_loc.columns:
        son5_col = "Haftalık Değişim (%)"

    if son5_col is not None:
        df_loc[son5_col] = pd.to_numeric(df_loc[son5_col], errors="coerce")
        mask &= df_loc[son5_col].between(son5_min, son5_max)

    # Filtre sonrası index'ler
    idx_filt = df_loc[mask].index
    if len(idx_filt) == 0:
        print("[Uyarı] KURAL_DIP_STRICT filtresine uyan hisse yok.")
        return pd.DataFrame()

    # JOIN: orijinal tablo üzerinden tüm kolonlar
    df_joined = df_multi.loc[idx_filt].copy()

    # EMA kolonunu çöz
    ema_col = _resolve_ema5_137_signal(df_joined)

    sort_by = []
    ascending = []

    if "FinalSkorPlus" in df_joined.columns:
        sort_by.append("FinalSkorPlus")
        ascending.append(False)

    if ema_col:
        sort_by.append(ema_col)
        ascending.append(False)

    sort_by += [
        "VolumeDeltaUp_Sort",
        "PUANLAMA_V4",
    ]
    ascending += [False, False]

    sort_by.append("FinalSkorEx")
    ascending.append(True)

    real_sort_by = [c for c in sort_by if c in df_joined.columns]
    real_ascending = [ascending[i] for i, c in enumerate(sort_by) if c in df_joined.columns]

    df_sorted = df_joined.sort_values(by=real_sort_by, ascending=real_ascending).reset_index(drop=True)

    df_sel = df_sorted.head(top_n).copy()
    df_sel["Kural"] = "KURAL_DIP_STRICT"
    df_sel["Kural_Adi"] = (
        f"Dip<=10, ATH uzak>=30, Aylık<=20, Son5g (0,15]; "
        f"FinalSkorPlus(desc, ops) > {ema_col or 'EMA5_137?(yok)'}(desc) > "
        "VolDelta(desc) > PUANLAMA_V4(desc) > FinalSkorEx(asc)"
    )
    return df_sel


def _build_merged_sheet(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df_all = pd.concat(dfs, ignore_index=True)
    grp = df_all.groupby("Sembol")

    rows: List[Dict[str, Any]] = []
    for sembol, g in grp:
        kurallar = sorted(set(g["Kural"].astype(str)))
        kural_listesi = ",".join(kurallar)
        kural_sayisi = len(kurallar)

        base_row = g.iloc[-1].to_dict()
        base_row["Kural_Listesi"] = kural_listesi
        base_row["Kural_Sayisi"] = kural_sayisi
        base_row["KaynakReiter"] = base_row.get("Kural_Adi", "")
        rows.append(base_row)

    df_merged = pd.DataFrame(rows)

    sort_cols = ["Kural_Sayisi"]
    ascending = [False]

    for c, asc in [
        ("FinalSkorPlus", False),
        ("PUANLAMA_V4", False),
        ("FinalSkorEx", True),
        ("VolumeDeltaUp_Sort", False),
    ]:
        if c in df_merged.columns:
            sort_cols.append(c)
            ascending.append(asc)

    df_merged = df_merged.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return df_merged


def main():
    # Bugün varsa bugünün dosyasını (CSV öncelikli) kullan; yoksa en son dosya.
    df_multi, date_str = _load_multim_today_or_latest()

    # Kural seçimleri
    df_k1 = _select_kural1(df_multi, top_n=20)
    df_k2 = _select_kural2(df_multi, top_n=20)
    df_k3 = _select_kural3(df_multi, top_n=30)
    df_k4 = _select_kural4(df_multi, top_n=20)
    df_k5 = _select_kural5_finalplus_combo(df_multi, top_n=20)
    df_dip_strict = _select_kural_dip_strict(df_multi, top_n=20)

    # Birleşik çıktı (aktüel tarih ile)
    df_merged = _build_merged_sheet(
        [df_k1, df_k2, df_k3, df_k4, df_k5, df_dip_strict]
    )

    out_xlsx = os.path.join(BASE_DIR, f"BEST_AUTO_COMBO_LISTE_{date_str}.xlsx")
    out_csv = os.path.join(BASE_DIR, f"BEST_AUTO_COMBO_LISTE_{date_str}.csv")

    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            if not df_k1.empty:
                df_k1.to_excel(writer, index=False, sheet_name="KURAL1_Puan_FinalEx")
            if not df_k2.empty:
                df_k2.to_excel(writer, index=False, sheet_name="KURAL2_EMA_Puan")
            if not df_k3.empty:
                df_k3.to_excel(writer, index=False, sheet_name="KURAL3_Dip_EMA")
            if not df_k4.empty:
                df_k4.to_excel(writer, index=False, sheet_name="KURAL4_Puan_Vol")
            if not df_k5.empty:
                df_k5.to_excel(writer, index=False, sheet_name="KURAL5_FinalPlus_Combo")
            if not df_dip_strict.empty:
                df_dip_strict.to_excel(writer, index=False, sheet_name="KURAL_DIP_STRICT")
            df_merged.to_excel(writer, index=False, sheet_name="HEPSI_BIRLESIK")
        print(f"[ÇIKTI] Excel üretildi: {out_xlsx}")
    except ModuleNotFoundError:
        print("[Uyarı] openpyxl yüklü değil, Excel yazılamadı. 'pip install openpyxl' kurup tekrar deneyin.")

    # CSV her zaman üret
    df_merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ÇIKTI] Birleşik liste (CSV): {out_csv}")


if __name__ == "__main__":
    main()