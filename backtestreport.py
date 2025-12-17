import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# AYARLAR
# ============================================================
MAX_HORIZON = 14          # 1..14 iş günü sonrası
WINDOWS = [3, 5, 14]      # özet pencereleri
TARGET_T_MINUS_1 = True   # her akşam: T-1 listelerini backtest et

# multim4 fiyat kaynağı
MULTIM_RE = re.compile(r"^multim4_(?P<date>\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)

# Kullanıcı listeleri (tam senin verdiğin etiketler)
LIST_SPECS = [
    # Ana listeler (backtest5.py'den)
    (re.compile(r"^LISTE5_2_5GUN_YUKSEK_RISK_TOP20_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE), "LISTE5_YUKSEK_RISK_5G", "csv"),
    (re.compile(r"^LISTE5_3_15GUN_DUSUK_RISK_MIN20_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE), "LISTE5_DUSUK_RISK_15G", "csv"),
    (re.compile(r"^RISKLI5_5GUN_VE_15GUN_BIRLESIK_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "RISKLI5_BIRLESIK", "xlsx"),

    # Yeni Avcı Stratejileri
    (re.compile(r"^DIP_AVCISI_LISTE_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "DIP_AVCISI", "xlsx"),
    (re.compile(r"^SIKISAN_PATLAMA_LISTESI_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "SIKISAN_PATLAMA", "xlsx"),

    # Diğer Yardımcı Listeler
    (re.compile(r"^BEST_AUTO_COMBO_LISTE_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "BEST_AUTO_COMBO", "xlsx"),
    (re.compile(r"^TEKNIK_TABANLI_SECIM_nextday5_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "TEKNIK_TABANLI", "xlsx"),
    (re.compile(r"^YARIN5_TEKNIK_LISTE_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "YARIN5_TEKNIK", "xlsx"),
    (re.compile(r"^BEST_RANK_FROM_MULTIM_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "BEST_RANK_MULTIM", "xlsx"),
    (re.compile(r"^PATTERN_NEXTDAY5_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE), "PATTERN_NEXTDAY5", "xlsx"),
]

# multim4 kolonları
COL_SYMBOL = "Sembol"
COL_DATE = "Tarih"
COL_CLOSE = "Fiyat (Son)"

# Liste dosyalarında sembol kolonunun olası aliasları
SYMBOL_ALIASES = [
    "Sembol", "SEMBOL", "Symbol", "SYMBOL", "Hisse", "HISSE", "Kod", "KOD", "Ticker", "TICKER"
]


# ============================================================
# Yardımcılar
# ============================================================

def _today_ts() -> pd.Timestamp:
    return pd.to_datetime(datetime.now().strftime("%Y-%m-%d")).normalize()


def _safe_sheet_name(name: str) -> str:
    name = re.sub(r"[:\\/?*\[\]]", "_", name).strip()
    if len(name) > 31:
        name = name[:31]
    return name or "SHEET"


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1254", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def _to_pct(series: pd.Series) -> pd.Series:
    # 0.0123 -> 1.23
    return pd.to_numeric(series, errors="coerce") * 100.0


def _resolve_symbol_col(df: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns]
    lower_map = {str(c).strip().lower(): str(c).strip() for c in cols}
    for a in SYMBOL_ALIASES:
        key = a.strip().lower()
        if key in lower_map:
            return lower_map[key]
    for c in cols:
        cl = c.lower()
        if "sembol" in cl or "symbol" in cl or "ticker" in cl or (cl in ["kod", "code"]):
            return c
    return None


# ============================================================
# Fiyat Paneli (multim4 serisinden)
# ============================================================

def _detect_multim_dates() -> List[str]:
    ds = []
    for fn in os.listdir(BASE_DIR):
        m = MULTIM_RE.match(fn)
        if m:
            ds.append(m.group("date"))
    return sorted(set(ds))


def _load_multim(date_str: str) -> Optional[pd.DataFrame]:
    path = os.path.join(BASE_DIR, f"multim4_{date_str}.csv")
    if not os.path.isfile(path):
        return None

    df = _read_csv_any(path)
    if COL_SYMBOL not in df.columns or COL_DATE not in df.columns or COL_CLOSE not in df.columns:
        return None

    df = df[[COL_SYMBOL, COL_DATE, COL_CLOSE]].copy()
    df[COL_SYMBOL] = df[COL_SYMBOL].astype(str).str.strip().str.upper()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.normalize()
    df[COL_CLOSE] = pd.to_numeric(df[COL_CLOSE], errors="coerce")
    df = df.dropna(subset=[COL_SYMBOL, COL_DATE, COL_CLOSE])
    df = df.sort_values([COL_SYMBOL, COL_DATE]).drop_duplicates([COL_SYMBOL, COL_DATE], keep="last")
    return df


def build_price_panel_from_multim() -> pd.DataFrame:
    dates = _detect_multim_dates()
    frames = []
    for d in dates:
        part = _load_multim(d)
        if part is not None and not part.empty:
            frames.append(part)

    if not frames:
        raise RuntimeError("multim4_*.csv dosyalarından fiyat paneli kurulamadı.")

    allp = pd.concat(frames, ignore_index=True)
    panel = allp.pivot_table(index=COL_SYMBOL, columns=COL_DATE, values=COL_CLOSE, aggfunc="last")
    panel = panel.sort_index(axis=1)
    return panel


# ============================================================
# Liste Dosyalarını Bul / Oku (sadece LIST_SPECS)
# ============================================================

def find_list_files_for_date(date_str: str) -> List[Tuple[str, str, str, str]]:
    """Returns list of tuples: (full_path, label, ext, filename)"""
    out = []
    for fn in os.listdir(BASE_DIR):
        for rx, label, ext in LIST_SPECS:
            m = rx.match(fn)
            if not m:
                continue
            d = m.group(1)
            if d != date_str:
                continue
            out.append((os.path.join(BASE_DIR, fn), label, ext, fn))

    # dedup by label
    dedup: Dict[str, Tuple[str, str, str, str]] = {}
    for rec in out:
        dedup[rec[1]] = rec
    return list(dedup.values())


def read_symbols_from_list_file(path: str, ext: str) -> List[str]:
    symbols: List[str] = []

    if ext == "csv":
        df = _read_csv_any(path)
        df.columns = [str(c).strip() for c in df.columns]
        sym_col = _resolve_symbol_col(df)
        if sym_col is None:
            return []
        s = df[sym_col].dropna().astype(str).str.strip().str.upper()
        symbols.extend(s.tolist())

    elif ext == "xlsx":
        import openpyxl  # noqa: F401

        xls = pd.ExcelFile(path, engine="openpyxl")
        for sh in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sh)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            sym_col = _resolve_symbol_col(df)
            if sym_col is None:
                continue
            s = df[sym_col].dropna().astype(str).str.strip().str.upper()
            symbols.extend(s.tolist())

    symbols = [x for x in symbols if x and x != "NAN"]
    return sorted(set(symbols))


# ============================================================
# Getiri Hesapları (iş günü bazlı)
# ============================================================

def compute_forward_returns(panel: pd.DataFrame, symbols: List[str], signal_date: pd.Timestamp) -> pd.DataFrame:
    cols = list(panel.columns)
    if signal_date not in cols:
        return pd.DataFrame()

    t0 = cols.index(signal_date)
    sub = panel.reindex(symbols)

    entry = sub.iloc[:, t0]
    out = pd.DataFrame({"Sembol": symbols})
    out["signal_date"] = signal_date
    out["entry_close"] = entry.values

    for h in range(1, MAX_HORIZON + 1):
        t = t0 + h
        if t >= len(cols):
            out[f"ret_{h}d"] = pd.NA
            out[f"px_{h}d"] = pd.NA
            continue
        px = sub.iloc[:, t]
        out[f"px_{h}d"] = px.values
        out[f"ret_{h}d"] = (px.values / entry.values) - 1.0

    return out


def add_window_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k in WINDOWS:
        ret_cols = [f"ret_{i}d" for i in range(1, k + 1)]
        arr = out[ret_cols].apply(pd.to_numeric, errors="coerce")

        out[f"avg_ret_1_{k}"] = arr.mean(axis=1, skipna=True)
        out[f"best_ret_1_{k}"] = arr.max(axis=1, skipna=True)
        out[f"worst_ret_1_{k}"] = arr.min(axis=1, skipna=True)
        out[f"hit_any_1_{k}"] = arr.gt(0).any(axis=1).astype(int)
        out[f"available_days_1_{k}"] = arr.notna().sum(axis=1)

    return out


# ============================================================
# Rapor (Türkçe / anlaşılır)
# ============================================================

def make_pretty_details(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["Sinyal Tarihi"] = pd.to_datetime(d["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d["Giriş Fiyatı (Kapanış)"] = pd.to_numeric(d["entry_close"], errors="coerce")
    d["1G Sonrası Fiyat"] = pd.to_numeric(d.get("px_1d"), errors="coerce")
    d["1G Getiri (%)"] = _to_pct(d.get("ret_1d")).round(2)

    for k in WINDOWS:
        d[f"1-{k}G Ortalama Getiri (%)"] = _to_pct(d.get(f"avg_ret_1_{k}")).round(2)
        d[f"1-{k}G En İyi Getiri (%)"] = _to_pct(d.get(f"best_ret_1_{k}")).round(2)
        d[f"1-{k}G En Kötü Getiri (%)"] = _to_pct(d.get(f"worst_ret_1_{k}")).round(2)
        d[f"1-{k}G En Az 1 Gün Pozitif mi?"] = pd.to_numeric(d.get(f"hit_any_1_{k}"), errors="coerce").fillna(0).astype(int)
        d[f"1-{k}G Veri Gün Sayısı"] = pd.to_numeric(d.get(f"available_days_1_{k}"), errors="coerce").fillna(0).astype(int)

    keep = [
        "Liste Adı",
        "Sinyal Tarihi",
        "Sembol",
        "Giriş Fiyatı (Kapanış)",
        "1G Sonrası Fiyat",
        "1G Getiri (%)",
        "Kaynak Dosya",
    ]
    for k in WINDOWS:
        keep += [
            f"1-{k}G Ortalama Getiri (%)",
            f"1-{k}G En İyi Getiri (%)",
            f"1-{k}G En Kötü Getiri (%)",
            f"1-{k}G En Az 1 Gün Pozitif mi?",
            f"1-{k}G Veri Gün Sayısı",
        ]

    keep = [c for c in keep if c in d.columns]
    d = d[keep].copy()
    return d


def build_scoreboard(df_all_raw: pd.DataFrame) -> pd.DataFrame:
    if df_all_raw.empty:
        return pd.DataFrame()

    rows = []
    for (label, sig_date), g in df_all_raw.groupby(["label", "signal_date"]):
        rec: Dict[str, object] = {
            "Liste Adı": label,
            "Sinyal Tarihi": pd.to_datetime(sig_date).strftime("%Y-%m-%d"),
            "Hisse Sayısı": int(len(g)),
        }

        r1 = pd.to_numeric(g.get("ret_1d"), errors="coerce")
        rec["Ort_Getiri_1G_%"] = float((r1.mean() * 100.0)) if r1.notna().any() else np.nan
        rec["Basari_Orani_1G_%"] = float((r1.gt(0).mean() * 100.0)) if r1.notna().any() else np.nan

        for k in WINDOWS:
            a = pd.to_numeric(g.get(f"avg_ret_1_{k}"), errors="coerce")
            b = pd.to_numeric(g.get(f"best_ret_1_{k}"), errors="coerce")
            h = pd.to_numeric(g.get(f"hit_any_1_{k}"), errors="coerce")
            avail = pd.to_numeric(g.get(f"available_days_1_{k}"), errors="coerce")

            rec[f"Ort_Getiri_1_{k}G_%"] = float((a.mean() * 100.0)) if a.notna().any() else np.nan
            rec[f"EnIyiGun_Ort_1_{k}G_%"] = float((b.mean() * 100.0)) if b.notna().any() else np.nan
            rec[f"EnAz1GunPozitif_1_{k}G_%"] = float((h.mean() * 100.0)) if h.notna().any() else np.nan
            rec[f"Ortalama_VeriGun_1_{k}G"] = float(avail.mean()) if avail.notna().any() else np.nan

        rows.append(rec)

    sb = pd.DataFrame(rows)

    for c in sb.columns:
        if c.endswith("_%"):
            sb[c] = pd.to_numeric(sb[c], errors="coerce").round(2)
        if c.endswith("_G"):
            sb[c] = pd.to_numeric(sb[c], errors="coerce").round(2)

    sort_cols = [c for c in ["Ort_Getiri_1G_%", "Ort_Getiri_1_5G_%"] if c in sb.columns]
    if sort_cols:
        sb = sb.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    return sb.reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    today = _today_ts()
    target_signal_date = today - pd.Timedelta(days=1) if TARGET_T_MINUS_1 else today
    target_str = target_signal_date.strftime("%Y-%m-%d")

    list_files = find_list_files_for_date(target_str)
    if not list_files:
        print(f"[Bilgi] T-1 listeleri bulunamadı: {target_str} (haftasonu/tatil olabilir).")
        return

    panel = build_price_panel_from_multim()
    if target_signal_date not in panel.columns:
        print(f"[Uyarı] multim4 fiyat panelinde sinyal günü yok: {target_str}")
        return

    all_raw_parts: List[pd.DataFrame] = []
    pretty_by_list: Dict[str, pd.DataFrame] = {}

    for full_path, label, ext, fname in list_files:
        syms = read_symbols_from_list_file(full_path, ext)
        if not syms:
            print(f"[Uyarı] Liste boş veya 'Sembol' bulunamadı: {fname} ({label})")
            continue

        raw = compute_forward_returns(panel, syms, target_signal_date)
        if raw.empty:
            print(f"[Uyarı] Fiyat bulunamadı veya hesap yapılamadı: {fname} ({label})")
            continue

        raw["label"] = label
        raw["source_file"] = fname
        raw = add_window_metrics(raw)

        all_raw_parts.append(raw)

        tmp = raw.copy()
        tmp["Liste Adı"] = label
        tmp["Kaynak Dosya"] = fname
        pretty_by_list[label] = make_pretty_details(tmp)

    if not all_raw_parts:
        print("[Uyarı] Hiçbir listeden getiri hesaplanamadı.")
        return

    df_all_raw = pd.concat(all_raw_parts, ignore_index=True)

    scoreboard = build_scoreboard(df_all_raw)

    tmp_all = df_all_raw.copy()
    tmp_all["Liste Adı"] = tmp_all["label"]
    tmp_all["Kaynak Dosya"] = tmp_all["source_file"]
    all_pretty = make_pretty_details(tmp_all)

    out_path = os.path.join(BASE_DIR, f"LIST_BACKTEST_RAPORU_{today.strftime('%Y-%m-%d')}.xlsx")

    import openpyxl  # noqa: F401
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        scoreboard.to_excel(writer, index=False, sheet_name="OZET_SKORBOARD")
        all_pretty.to_excel(writer, index=False, sheet_name="TUM_DETAY")

        for label, dfp in pretty_by_list.items():
            sh = _safe_sheet_name("LISTE__" + label)
            dfp.to_excel(writer, index=False, sheet_name=sh)

    print(f"[ÇIKTI] Liste backtest raporu üretildi: {out_path}")


if __name__ == "__main__":
    main()
