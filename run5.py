import os
import sys
import subprocess
from datetime import datetime

def run_script(script_name: str):
    """Belirtilen Python betiğini çalıştırır ve hataları kontrol eder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)

    if not os.path.isfile(script_path):
        print(f"[Hata] {script_name} bulunamadı: {script_path}")
        # Hata durumunda diğer betiklerin çalışmasını engellemek için dur.
        sys.exit(1)

    print(f"\n[ÇALIŞTIRILIYOR] {script_name}")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=base_dir,
    )

    if result.returncode != 0:
        print(f"[Hata] {script_name} hatayla bitti. (returncode={result.returncode})")
        # Hata durumunda diğer betiklerin çalışmasını engellemek için dur.
        sys.exit(result.returncode)

    print(f"[BİTTİ] {script_name} başarıyla tamamlandı.\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=====================================================")
    print("===            YATIRIMCI ASİSTANI v5.1            ===")
    print("=====================================================")

    # ADIM 1: Temel veri üretimi ve geçmiş performans analizi.
    # Bu adım, diğer tüm adımların temelini oluşturur.
    print("\n--- ADIM 1: Geçmiş Performans Analizi Başlatılıyor ---")
    run_script("backtest5.py")
    run_script("stratejidogrulama5.py")

    # ADIM 2: Güncel verilerle (bugünün multim4'ü) hisse seçim listeleri üretimi.
    # Bu adımlar birbirine bağlı değildir, paralel olarak düşünülebilir.
    print("\n--- ADIM 2: Günlük Hisse Seçim Stratejileri Çalıştırılıyor ---")
    run_script("kural2.py")
    run_script("pattern_nextday5.py")
    run_script("dipavcisi5.py")
    run_script("sikisanpatlama5.py")
    run_script("nextday5.py")
    run_script("teknik_grup_nextday5.py")
    run_script("teknik_tabanli_secim5.py")
    run_script("today.py")
    run_script("best_rank_from_multim.py") # Artık doğru şekilde çağrılıyor.

    # ADIM 3: Üretilen tüm listeleri tek bir raporda birleştirme.
    print("\n--- ADIM 3: Tüm Seçim Listeleri Birleştiriliyor ---")
    merge_script = os.path.join(base_dir, "merge_all_selections.py")
    if os.path.isfile(merge_script):
        run_script("merge_all_selections.py")

        today_str = datetime.now().strftime("%Y-%m-%d")
        source_name = f"BIRLESIK_TUM_LISTELER_{today_str}.xlsx"
        source_path = os.path.join(base_dir, source_name)
        target_name = f"birlesmisdosya_{today_str}.xlsx"
        target_path = os.path.join(base_dir, target_name)

        if os.path.isfile(source_path):
            try:
                if os.path.isfile(target_path):
                    os.remove(target_path)
                # os.replace yerine os.rename kullanmak bazı sistemlerde daha güvenilirdir.
                os.rename(source_path, target_path)
                print(f"[Bilgi] Birleşik dosya yeniden adlandırıldı: {target_name}")
            except Exception as e:
                print(f"[Hata] Birleşik dosya yeniden adlandırılamadı: {e}")
        else:
            # Bu bir hata değil, sadece merge_all_selections bir çıktı üretmemiş olabilir.
            print(f"[Uyarı] Beklenen birleşik dosya bulunamadı: {source_name}")
    else:
        print("[Uyarı] merge_all_selections.py bulunamadı, birleştirme yapılmadı.")

    print(
        "\n[TAMAMLANDI] Tüm analiz ve seçim akışı başarıyla tamamlandı."
        f"\nSonuç dosyası: birlesmisdosya_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    )


if __name__ == "__main__":
    main()
