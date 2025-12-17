import os
import sys
from datetime import datetime, timedelta

import pandas as pd


# ===================== Yardımcılar =====================

def find_backtest5_file(output_dir: str, date_str: str) -> str:
    """
    GERCEK_BACKTEST5_TO_LATEST_{date}.csv/xlsx dosyasını bulur.
    """
    base = f"GERCEK_BACKTEST5_TO_LATEST_{date_str}"
    csv_path = os.path.join(output_dir, base + ".csv")
    xlsx_path = os.path.join(output_dir, base + ".xlsx")

    if os.path.isfile(csv_path):
        return csv_path
    if os.path.isfile(xlsx_path):
        return xlsx_path
    raise ValueError(
        f"Backtest5 dosyası bulunamadı (ne CSV ne XLSX): {csv_path} / {xlsx_path}"
    )


def read_backtest_any(back_path: str) -> pd.DataFrame:
    ext = os.path.splitext(back_path.lower())[1]
    if ext == ".csv":
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(back_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "csv", b"", 0, 1, "Verilen encoding denemeleri başarısız."
        )
    elif ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(back_path, engine="openpyxl")
        except ImportError:
            raise RuntimeError(
                "openpyxl paketi yüklü değil. 'pip install openpyxl' ile kurun."
            )
    else:
        raise ValueError(f"Desteklenmeyen uzantı: {ext}")


def coerce_numeric(df: pd.DataFrame, min_non_null_ratio: float = 0.5) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            non_null_ratio = df[col].notna().mean()
            if non_null_ratio >= min_non_null_ratio:
                col_lower = str(col).lower()
                if "sembol" in col_lower or "kod" in col_lower:
                    continue
                converted = pd.to_numeric(df[col], errors="ignore")
                if converted.dtype != object:
                    df[col] = converted
    return df


def _load_multim4_for_date(base_dir: str, date_str: str) -> pd.DataFrame | None:
    """Belirli tarih için multim4_YYYY-MM-DD.csv dosyasını oku."""
    multi_name = f"multim4_{date_str}.csv"
    multi_path = os.path.join(base_dir, multi_name)
    if not os.path.isfile(multi_path):
        return None
    try:
        df_multi = pd.read_csv(multi_path)
        return df_multi
    except Exception:
        return None


def merge_latest_prices(base_dir: str, back_date_str: str, df_back: pd.DataFrame) -> pd.DataFrame:
    """
    backtest tarihi (back_date_str) için:
      - multim4_back:   dünkü/güncel fiyat (Fiyat_Guncel, Son_7_Gun_%, Son_30_Gun_%)
      - multim4_(back_date-1): Fiyat_Dun
    üzerinden:
      * Fiyat_Guncel
      * Fiyat_Dun
      * Son_Gun_% = (Fiyat_Guncel - Fiyat_Dun) / Fiyat_Dun * 100
      * Son_7_Gun_%
      * Son_30_Gun_%
    kolonlarını ekler.
    """
    # ---- BUGÜNÜN(multim4_back_date) DOSYASI ----
    df_multi_today = _load_multim4_for_date(base_dir, back_date_str)
    if df_multi_today is None:
        print(f"[Uyarı] (nextday5) Güncel fiyat dosyası bulunamadı: multim4_{back_date_str}.csv")
        df_back["Fiyat_Guncel"] = pd.NA
        df_back["Fiyat_Dun"] = pd.NA
        df_back["Son_Gun_%"] = pd.NA
        df_back["Son_7_Gun_%"] = pd.NA
        df_back["Son_30_Gun_%"] = pd.NA
        return df_back

    print(f"[Bilgi] (nextday5) Güncel fiyat dosyası kullanılıyor: multim4_{back_date_str}.csv")

    if "Sembol" not in df_multi_today.columns:
        print("[Uyarı] (nextday5) multim4 dosyasında 'Sembol' kolonu yok.")
        df_back["Fiyat_Guncel"] = pd.NA
        df_back["Fiyat_Dun"] = pd.NA
        df_back["Son_Gun_%"] = pd.NA
        df_back["Son_7_Gun_%"] = pd.NA
        df_back["Son_30_Gun_%"] = pd.NA
        return df_back

    price_col = "Fiyat (Son)"
    weekly_col = "Haftalık Değişim (%)"
    monthly_col = "Aylık Değişim (%)"

    use_cols = ["Sembol"]
    if price_col in df_multi_today.columns:
        use_cols.append(price_col)
    if weekly_col in df_multi_today.columns:
        use_cols.append(weekly_col)
    if monthly_col in df_multi_today.columns:
        use_cols.append(monthly_col)

    df_today_small = df_multi_today[use_cols].copy()

    rename_map_today = {}
    if price_col in df_today_small.columns:
        rename_map_today[price_col] = "Fiyat_Guncel"
    if weekly_col in df_today_small.columns:
        rename_map_today[weekly_col] = "Son_7_Gun_%"
    if monthly_col in df_today_small.columns:
        rename_map_today[monthly_col] = "Son_30_Gun_%"

    df_today_small = df_today_small.rename(columns=rename_map_today)
    df_today_small = df_today_small.drop_duplicates("Sembol", keep="last")

    df_merged = df_back.merge(df_today_small, on="Sembol", how="left")

    # ---- DÜNKÜ multim4 (Fiyat_Dun) ----
    try:
        back_date = datetime.strptime(back_date_str, "%Y-%m-%d").date()
        yesterday_date = back_date - timedelta(days=1)
        yesterday_str = yesterday_date.strftime("%Y-%m-%d")
    except Exception:
        yesterday_str = back_date_str

    df_multi_yesterday = _load_multim4_for_date(base_dir, yesterday_str)
    if df_multi_yesterday is not None and "Sembol" in df_multi_yesterday.columns:
        use_cols_y = ["Sembol"]
        if price_col in df_multi_yesterday.columns:
            use_cols_y.append(price_col)
        df_yest_small = df_multi_yesterday[use_cols_y].copy()
        if price_col in df_yest_small.columns:
            df_yest_small = df_yest_small.rename(columns={price_col: "Fiyat_Dun"})
        df_yest_small = df_yest_small.drop_duplicates("Sembol", keep="last")
        df_merged = df_merged.merge(df_yest_small, on="Sembol", how="left")
    else:
        df_merged["Fiyat_Dun"] = pd.NA

    # ---- Sayısal dönüşüm ve Son_Gun_% hesabı ----
    for col in ["Fiyat_Guncel", "Fiyat_Dun", "Son_7_Gun_%", "Son_30_Gun_%"]:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    if "Fiyat_Guncel" in df_merged.columns and "Fiyat_Dun" in df_merged.columns:
        df_merged["Son_Gun_%"] = (
            (df_merged["Fiyat_Guncel"] - df_merged["Fiyat_Dun"])
            / df_merged["Fiyat_Dun"]
            * 100.0
        )
    else:
        df_merged["Son_Gun_%"] = pd.NA

    return df_merged


def run_selection(df_back: pd.DataFrame, base_dir: str):
    print("[Bilgi] (nextday5) Yeni skorlama mantığı ile seçim yapılıyor (5 gün + 15 gün).")

    required_short = [
        "Sembol",
        "Max_Getiri_%",
        "Getiri_1Hafta_Sonu_%",
        "Zirve_Kayip_%",
    ]
    missing_short = [c for c in required_short if c not in df_back.columns]

    required_mid = [
        "Sembol",
        "Max_Getiri_15Gun_%",
        "Getiri_15Gun_Sonu_%",
        "Zirve_Kayip_15Gun_%",
    ]
    missing_mid = [c for c in required_mid if c not in df_back.columns]

    if missing_short:
        print(f"[Uyarı] Kısa vade için gerekli bazı kolonlar yok: {missing_short}")
    if missing_mid:
        print(f"[Uyarı] 15 günlük orta vade için gerekli bazı kolonlar yok: {missing_mid}")

    if missing_short and missing_mid:
        print("[Hata] Ne kısa vade ne orta vade için yeterli kolon yok. Çıkılıyor.")
        return

    numeric_cols = []
    if not missing_short:
        numeric_cols.extend(["Max_Getiri_%", "Getiri_1Hafta_Sonu_%", "Zirve_Kayip_%"])
    if not missing_mid:
        numeric_cols.extend(["Max_Getiri_15Gun_%", "Getiri_15Gun_Sonu_%", "Zirve_Kayip_15Gun_%"])

    for col in numeric_cols:
        if col in df_back.columns:
            df_back[col] = pd.to_numeric(df_back[col], errors="coerce")

    df_back.dropna(subset=["Sembol"], inplace=True)
    if df_back.empty:
        print("[Uyarı] Analiz edilecek geçerli veri bulunamadı.")
        return

    risk_faktoru = 0.5

    short_out = pd.DataFrame()
    mid_out = pd.DataFrame()

    # ----- 5 GÜN -----
    if not missing_short:
        df_short = df_back.dropna(subset=required_short).copy()
        if df_short.empty:
            print("[Uyarı] Kısa vade için geçerli satır bulunamadı.")
        else:
            df_short["Risk_Ayarlı_Skor_5"] = (
                df_short["Max_Getiri_%"] - (df_short["Zirve_Kayip_%"] * risk_faktoru)
            )
            df_short = df_short[df_short["Max_Getiri_%"] > 0].copy()

            df_best_per_symbol_short = df_short.sort_values(
                "Risk_Ayarlı_Skor_5", ascending=False
            ).drop_duplicates("Sembol")

            top_20_short = df_best_per_symbol_short.sort_values(
                "Risk_Ayarlı_Skor_5", ascending=False
            ).head(20)

            print("\n========== (nextday5) En İyi 20 Kısa Vade (5 Gün) ==========")

            report_cols_short = {
                "Sembol": "Sembol",
                "Fiyat (Son)": "Fiyat (Son)",          # backtest günü fiyatı
                "Fiyat_Dun": "Fiyat (Son)_Dun",
                "Fiyat_Guncel": "Fiyat (Son)_Bugun",   # bugünkü fiyat (multim4)
                "Son_Gun_%": "Son Gün %",
                "Son_7_Gun_%": "Son 7 Gün %",
                "Son_30_Gun_%": "Son 30 Gün %",
                "Risk_Ayarlı_Skor_5": "Skor_5Gun",
                "Max_Getiri_%": "Max Potansiyel 5Gün (%)",
                "Zirve_Kayip_%": "Zirve Kayıp 5Gün (%)",
                "Getiri_1Hafta_Sonu_%": "1 Hafta Sonu Getiri (%)",
                "Analiz_Tarihi_str": "Analiz Günü",
            }

            if top_20_short.empty:
                print("Kısa vade skorlamaya uygun hisse bulunamadı.")
            else:
                print(
                    top_20_short[report_cols_short.keys()]

                    .rename(columns=report_cols_short)
                    .to_string(index=False)
                )

            short_out = top_20_short[report_cols_short.keys()].rename(columns=report_cols_short)

    # ----- 15 GÜN -----
    if not missing_mid:
        df_mid = df_back.dropna(subset=required_mid).copy()
        if df_mid.empty:
            print("[Uyarı] Orta vade (15 gün) için geçerli satır bulunamadı.")
        else:
            df_mid["Risk_Ayarlı_Skor_15"] = (
                df_mid["Max_Getiri_15Gun_%"] - (df_mid["Zirve_Kayip_15Gun_%"] * risk_faktoru)
            )

            # BURADA EŞİK 30 -> 20
            df_mid = df_mid[df_mid["Max_Getiri_15Gun_%"] >= 20.0].copy()

            df_best_per_symbol_mid = df_mid.sort_values(
                "Risk_Ayarlı_Skor_15", ascending=False
            ).drop_duplicates("Sembol")

            top_20_mid = df_best_per_symbol_mid.sort_values(
                "Risk_Ayarlı_Skor_15", ascending=False
            ).head(20)

            print("\n========== (nextday5) En İyi 20 Orta Vade (15 Gün) ==========")

            report_cols_mid = {
                "Sembol": "Sembol",
                "Fiyat (Son)": "Fiyat (Son)",
                "Fiyat_Dun": "Fiyat (Son)_Dun",
                "Fiyat_Guncel": "Fiyat (Son)_Bugun",
                "Son_Gun_%": "Son Gün %",
                "Son_7_Gun_%": "Son 7 Gün %",
                "Son_30_Gun_%": "Son 30 Gün %",
                "Risk_Ayarlı_Skor_15": "Skor_15Gun",
                "Max_Getiri_15Gun_%": "Max Potansiyel 15Gün (%)",
                "Zirve_Kayip_15Gun_%": "Zirve Kayıp 15Gün (%)",
                "Getiri_15Gun_Sonu_%": "15 Gün Sonu Getiri (%)",
                "Analiz_Tarihi_str": "Analiz Günü",
            }

            if top_20_mid.empty:
                print("Orta vade (15 gün) skorlamaya uygun hisse bulunamadı.")
            else:
                print(
                    top_20_mid[report_cols_mid.keys()]
                    .rename(columns=report_cols_mid)
                    .to_string(index=False)
                )

            mid_out = top_20_mid[report_cols_mid.keys()].rename(columns=report_cols_mid)

    if short_out.empty and mid_out.empty:
        print("[Uyarı] (nextday5) Çıktı yok, Excel yazılmayacak.")
        return

    # --- SAYISAL KOLONLARI 2 ONDALIK BASAMAĞA YUVARLA ---
    def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if num_cols:
            df[num_cols] = df[num_cols].round(2)
        return df

    short_out = _round_numeric(short_out)
    mid_out = _round_numeric(mid_out)

    # ----- TEK EXCEL, VERSIYONLU ISIM -----
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    excel_path = os.path.join(
        base_dir,
        f"RISK_AYARLI_SECIM_ALL_nextday5_{today_str}.xlsx",
    )

    combined_list = []
    if not short_out.empty:
        tmp = short_out.copy()
        tmp["Horizon"] = "SHORT_5GUN"
        combined_list.append(tmp)
    if not mid_out.empty:
        tmp = mid_out.copy()
        tmp["Horizon"] = "MID_15GUN"
        combined_list.append(tmp)

    if combined_list:
        df_combined = pd.concat(combined_list, ignore_index=True)

        if "Skor_5Gun" not in df_combined.columns:
            df_combined["Skor_5Gun"] = 0.0
        if "Skor_15Gun" not in df_combined.columns:
            df_combined["Skor_15Gun"] = 0.0

        df_combined["Skor_5Gun"] = pd.to_numeric(df_combined["Skor_5Gun"], errors="coerce").fillna(0)
        df_combined["Skor_15Gun"] = pd.to_numeric(df_combined["Skor_15Gun"], errors="coerce").fillna(0)

        df_combined["Genel_Skor"] = df_combined[["Skor_5Gun", "Skor_15Gun"]].max(axis=1)

        if "Sembol" in df_combined.columns:
            df_best_combined = (
                df_combined
                .sort_values("Genel_Skor", ascending=False)
                .drop_duplicates(subset=["Sembol"])
            )
        else:
            df_best_combined = df_combined.copy()

        df_best_combined = df_best_combined.sort_values("Genel_Skor", ascending=False).reset_index(drop=True)
        df_best_combined = _round_numeric(df_best_combined)
    else:
        df_combined = pd.DataFrame()
        df_best_combined = pd.DataFrame()

    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            if not short_out.empty:
                short_out.to_excel(writer, sheet_name="SHORT_5GUN", index=False)
            if not mid_out.empty:
                mid_out.to_excel(writer, sheet_name="MID_15GUN", index=False)
            if not df_best_combined.empty:
                df_best_combined.to_excel(writer, sheet_name="KOMBI_EN_IYI", index=False)
        print(f"[ÇIKTI] (nextday5) Sonuçlar Excel'e yazıldı: {excel_path}")
    except ImportError:
        print("[Uyarı] openpyxl yok, Excel yazılamadı.")


def detect_latest_date5(output_dir: str):
    import re
    pattern = re.compile(
        r"GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.(csv|xlsx)$"
    )
    dates = []
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            dates.append(m.group(1))
    if not dates:
        return None
    return sorted(dates)[-1]


def find_latest_backtest5_any(output_dir: str):
    import re
    pattern = re.compile(
        r"GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.(csv|xlsx)$"
    )
    candidates = []
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            date_part = m.group(1)
            ext = m.group(2)
            candidates.append((date_part, ext, os.path.join(output_dir, fname)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    latest_date = candidates[-1][0]
    latest_same_date = [c for c in candidates if c[0] == latest_date]
    for c in latest_same_date:
        if c[1] == "csv":
            return c[2]
    return latest_same_date[0][2]


def main():
    global base_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = base_dir

    date_str = detect_latest_date5(output_dir)
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        back_path = find_backtest5_file(output_dir, date_str)
    except ValueError:
        back_path = find_latest_backtest5_any(output_dir)
        if back_path is None:
            print("[Hata] (nextday5) Hiç v5 backtest dosyası bulunamadı.")
            sys.exit(1)

    print(f"[Bilgi] (nextday5) Backtest5 dosyası seçildi: {back_path}")

    try:
        df_back = read_backtest_any(back_path)
    except Exception as e:
        print(f"[Hata] (nextday5) Backtest5 dosyası okunamadı: {e}")
        sys.exit(1)

    df_back = coerce_numeric(df_back)

    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(back_path))
    if m:
        back_date_str = m.group(1)
    else:
        back_date_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Güncel fiyat & son 1/7/30 gün değişimi ile birleştir
    df_back = merge_latest_prices(base_dir, back_date_str, df_back)

    run_selection(df_back, base_dir)

    print("\n[Tamamlandı] (nextday5) Günlük seçim işlemi tamamlandı.")


if __name__ == "__main__":
    main()