import os
import re
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np

# =================================================================
# Strateji Doğrulama ve Optimizasyon Motoru
# Amaç:   Belirlenen bir yatırım stratejisinin (örneğin "Sıkışan Patlama")
#         geçmiş verilerdeki (backtest5) performansını ölçmek ve
#         bu doğrulanmış stratejiyi bugünün verisine uygulamak.
# =================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKTEST5_PATTERN = r"^GERCEK_BACKTEST5_TO_LATEST_(\d{4}-\d{2}-\d{2})\.(csv|xlsx)$"
MULTIM4_PATTERN = r"^multim4_(\d{4}-\d{2}-\d{2})\.csv$"
OUTPUT_FILENAME_TEMPLATE = "DOGRULANMIS_STRATEJI_LISTESI_{date}.xlsx"


def _detect_latest_file(folder: str, pattern: str) -> Optional[Tuple[str, str]]:
    """Belirtilen pattern'e uyan en güncel dosyayı bulur."""
    rx = re.compile(pattern, re.IGNORECASE)
    candidates: list[tuple[date, str, str]] = []
    for fname in os.listdir(folder):
        m = rx.match(fname)
        if m:
            d_str = m.group(1)
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                candidates.append((d, os.path.join(folder, fname), fname))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    latest_date, latest_path, _ = candidates[-1]
    return latest_date.strftime("%Y-%m-%d"), latest_path


def _read_backtest_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Desteklenmeyen dosya uzantısı: {ext}")


# =================================================================
# ==              STRATEJİ TANIMLAMA BÖLÜMÜ                     ==
# =================================================================
# Buraya yeni stratejiler ekleyebilir veya mevcutları değiştirebilirsiniz.

def is_dip_avcisi_adayi(row: pd.Series) -> bool:
    """Bir hissenin 'Dip Avcısı' stratejisine uyup uymadığını kontrol eder."""
    try:
        # Aşama 1: Geniş Dip Havuzu
        is_in_dip_pool = (
            (row.get("Dipten Uzaklık (%)", 100) <= 15) and
            (row.get("ATH'ye Uzaklık (%)", 0) >= 50) and
            (row.get("Fiyat (Son)", 0) > 1.0)
        )
        if not is_in_dip_pool:
            return False

        # Aşama 2: Dönüş Emaresi Filtresi
        is_turning_around = (
            (30 <= row.get("RSI", 0) <= 55) and
            (row.get("MACD_Positive") == 1) and
            (row.get("Haftalık Değişim (%)", 0) > -5) and
            (row.get("Aylık Değişim (%)", 100) < 25)
        )
        if not is_turning_around:
            return False

        return True  # Tüm koşullar sağlandı
    except (TypeError, ValueError):
        return False


def is_sikisan_patlama_adayi(row: pd.Series) -> bool:
    """Bir hissenin 'Sıkışan Patlama' stratejisine uyup uymadığını kontrol eder."""
    try:
        # Genel Kalite
        if not (row.get("PUANLAMA_V4", 0) >= 2.0 and row.get("FinalSkorEx", 0) >= 0):
            return False

        # Sıkışma (Squeeze)
        # HATA DÜZELTMESİ: .between() yerine standart karşılaştırma kullanıldı.
        mask_sikisma = (row.get("BB_W", 100) <= 20) and (40 <= row.get("RSI", 0) <= 65)
        if not mask_sikisma:
            return False

        # Patlama Tetiği (Trigger)
        has_signal = row.get("Sinyal") == 100 or row.get("Confirmed_BUY") == 1
        has_volume = row.get("VolumeDeltaUp_Sort", 0) > 0

        if not (has_signal and has_volume):
            return False

        return True  # Tüm koşullar sağlandı
    except (TypeError, ValueError):
        return False

# Strateji sözlüğü: Buraya yeni stratejiler ekleyebilirsiniz.
STRATEGIES = {
    "Dip_Avcisi": is_dip_avcisi_adayi,
    "Sikisan_Patlama": is_sikisan_patlama_adayi,
}

def analyze_strategy_performance(df_backtest: pd.DataFrame, strategy_name: str, strategy_func) -> Dict[str, Any]:
    """Verilen stratejinin geçmiş performansını analiz eder."""
    print(f"\n[Analiz] '{strategy_name}' stratejisi doğrulanıyor...")
    
    # Stratejiye uyan geçmiş satırları bul
    df_backtest['StratejiyeUygun'] = df_backtest.apply(strategy_func, axis=1)
    past_success_rows = df_backtest[df_backtest['StratejiyeUygun']].copy()

    if past_success_rows.empty:
        print("  -> Sonuç: Geçmişte bu stratejiye uyan hiç pozisyon bulunamadı.")
        return {"name": strategy_name, "n_trades": 0, "avg_return_5d": 0, "avg_drawdown_5d": 0, "hit_rate": 0}

    # Performans Metriklerini Hesapla
    n_trades = len(past_success_rows)
    avg_return_5d = past_success_rows["Max_Getiri_%"].mean()
    avg_drawdown_5d = past_success_rows["Zirve_Kayip_%"].mean()
    hit_rate = (past_success_rows["Max_Getiri_%"] >= 12).mean() * 100
    
    print(f"  -> Sonuç: {n_trades} pozisyon bulundu. Ort. Getiri: %{avg_return_5d:.2f}, Başarı Oranı: %{hit_rate:.1f}")

    return {
        "name": strategy_name,
        "n_trades": n_trades,
        "avg_return_5d": avg_return_5d,
        "avg_drawdown_5d": avg_drawdown_5d,
        "hit_rate": hit_rate,
    }


def main():
    print("[ÇALIŞTIRILIYOR] stratejidogrulama5.py")

    # --- 1. Gerekli Dosyaları Bul ve Oku ---
    backtest_info = _detect_latest_file(BASE_DIR, BACKTEST5_PATTERN)
    if not backtest_info:
        print("[Hata] GERCEK_BACKTEST5 dosyası bulunamadı.")
        return
    _, backtest_path = backtest_info
    print(f"[Bilgi] Kullanılan backtest dosyası: {backtest_path}")
    df_backtest = _read_backtest_any(backtest_path)

    multim_info = _detect_latest_file(BASE_DIR, MULTIM4_PATTERN)
    if not multim_info:
        print("[Hata] Güncel multim4 dosyası bulunamadı.")
        return
    today_date, multim_path = multim_info
    print(f"[Bilgi] Kullanılan güncel multim4 dosyası: {multim_path}")
    df_today = pd.read_csv(multim_path, low_memory=False)

    # --- 2. Tüm Stratejileri Doğrula ---
    all_perf_metrics = []
    for name, func in STRATEGIES.items():
        # Her strateji için gerekli olabilecek tüm kolonları hazırla
        all_possible_cols = [
            "PUANLAMA_V4", "FinalSkorEx", "BB_W", "RSI", "Sinyal", "Confirmed_BUY", 
            "VolumeDeltaUp_Sort", "Dipten Uzaklık (%)", "ATH'ye Uzaklık (%)", 
            "Fiyat (Son)", "MACD_Positive", "Haftalık Değişim (%)", "Aylık Değişim (%)",
            "Max_Getiri_%", "Zirve_Kayip_%"
        ]
        for df in [df_backtest, df_today]:
            for col in all_possible_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = np.nan # Eksikse NaN ile doldur

        perf = analyze_strategy_performance(df_backtest.copy(), name, func)
        all_perf_metrics.append(perf)
    
    karne_df = pd.DataFrame(all_perf_metrics)

    # --- 3. Stratejileri Bugünün Hisselerine Uygula ---
    all_candidates = []
    for name, func in STRATEGIES.items():
        df_today['StratejiyeUygun'] = df_today.apply(func, axis=1)
        todays_candidates = df_today[df_today['StratejiyeUygun']].copy()
        if not todays_candidates.empty:
            todays_candidates['Strateji'] = name
            all_candidates.append(todays_candidates)

    if not all_candidates:
        print("\n[Bilgi] Bugün hiçbir stratejiye uyan hisse bulunamadı.")
        return
        
    final_df = pd.concat(all_candidates, ignore_index=True)

    # --- 4. Sonuçları Kaydet ---
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME_TEMPLATE.format(date=today_date))
    try:
        import openpyxl
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="BUGUNUN_ADAYLARI", index=False)
            karne_df.to_excel(writer, sheet_name="STRATEJI_KARNESI", index=False)
        print(f"\n[ÇIKTI] Doğrulanmış Strateji listesi ve karnesi oluşturuldu: {output_path}")
    except ImportError:
        print("\n[Uyarı] 'openpyxl' paketi kurulu değil. Excel çıktısı oluşturulamadı.")


if __name__ == "__main__":
    main()