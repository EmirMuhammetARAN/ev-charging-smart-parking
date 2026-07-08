"""
Dataset İndirme Script'i
========================
Roboflow SDK kullanarak EV şarj istasyonu dataset'lerini YOLOv8 formatında indirir.

Kullanım:
    python scripts/download_datasets.py

Gereksinimler:
    - .env dosyasında ROBOFLOW_API_KEY tanımlı olmalı
    - pip install roboflow python-dotenv
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Proje kök dizinini belirle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .env dosyasını yükle
load_dotenv(PROJECT_ROOT / ".env")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


# ============================================================
# İndirilecek Dataset'ler
# ============================================================
# Her dataset için: (workspace, project, version, hedef_klasör)
# NOT: Bu bilgiler Roboflow'daki URL'den alınır.
#   Örn: https://universe.roboflow.com/evchargeaero/ev-xxxxx/dataset/1
#        workspace = "evchargeaero", project = "ev-xxxxx", version = 1
#
# Aşağıdaki değerler varsayılan tahminlerdir.
# Roboflow'da dataset sayfasına git → "Use this Dataset" → "Download" sekmesinde
# doğru workspace/project/version bilgisini görebilirsin.
# ============================================================

DATASETS = [
    {
        "name": "evchargeaero",
        "description": "EV by evchargeaero — 176 img — car, charging cable, charging socket",
        "workspace": "evchargeaero",
        "project": "ev-2cusp",
        "version": 1,
        "target_dir": "evchargeaero",
    },
    {
        "name": "ev_charging_10k",
        "description": "EV Charging — 10k img — Car, Person, Socket",
        "workspace": "project-gtoq5",
        "project": "ev-charging",
        "version": 15,
        "target_dir": "ev_charging_10k",
    },
    {
        "name": "ev_charging_471",
        "description": "EV Charging — 471 img — CCS1AC, CCS1DC, CCS2DC, Charging, Socket",
        "workspace": "object-detection-4crhd",
        "project": "ev-charging-t81wn",
        "version": 1,
        "target_dir": "ev_charging_471",
    },
    {
        "name": "parking_lot",
        "description": "Parking lot occupancy — 8.5k img — 0 (empty), 1 (occupied)",
        "workspace": "team-mxd1n",
        "project": "parking-lot-occupancy-detection",
        "version": 1,
        "target_dir": "parking_lot",
    },
]


def check_api_key():
    """API key'in tanımlı olup olmadığını kontrol et."""
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "your_api_key_here":
        print("=" * 60)
        print("❌ HATA: Roboflow API key tanımlı değil!")
        print()
        print("Çözüm:")
        print("  1. https://roboflow.com adresinden ücretsiz hesap aç")
        print("  2. Settings > Roboflow API Key kısmından key'ini kopyala")
        print("  3. .env dosyasını aç ve key'i yapıştır:")
        print(f"     Dosya: {PROJECT_ROOT / '.env'}")
        print("     ROBOFLOW_API_KEY=rf_xxxxxxxxxxxxxxxxx")
        print("=" * 60)
        sys.exit(1)
    print(f"✅ API key bulundu: {ROBOFLOW_API_KEY[:8]}...")


def download_dataset(dataset_info: dict):
    """Tek bir dataset'i Roboflow'dan indir."""
    from roboflow import Roboflow

    name = dataset_info["name"]
    target = RAW_DIR / dataset_info["target_dir"]

    # Zaten indirilmiş mi kontrol et
    if target.exists() and any(target.iterdir()):
        print(f"⏭️  {name}: Zaten indirilmiş, atlanıyor ({target})")
        return True

    print(f"\n{'='*60}")
    print(f"📥 İndiriliyor: {name}")
    print(f"   Açıklama: {dataset_info['description']}")
    print(f"   Workspace: {dataset_info['workspace']}")
    print(f"   Project: {dataset_info['project']}")
    print(f"   Version: {dataset_info['version']}")
    print(f"   Hedef: {target}")
    print(f"{'='*60}")

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
        version = project.version(dataset_info["version"])

        # YOLOv8 formatında indir
        dataset = version.download(
            model_format="yolov8",
            location=str(target),
            overwrite=False,
        )

        print(f"✅ {name}: Başarıyla indirildi!")
        print(f"   Konum: {target}")

        # İndirilen data.yaml'ı göster
        yaml_path = target / "data.yaml"
        if yaml_path.exists():
            print(f"   Config: {yaml_path}")
            with open(yaml_path, "r", encoding="utf-8") as f:
                print(f"   İçerik:\n{f.read()}")

        return True

    except Exception as e:
        print(f"❌ {name}: İndirme hatası!")
        print(f"   Hata: {e}")
        print()
        print(f"   Olası çözümler:")
        print(f"   1. Roboflow'da dataset sayfasına git")
        print(f"   2. 'Use this Dataset' > 'Download' sekmesini aç")
        print(f"   3. Doğru workspace/project/version bilgisini kontrol et")
        print(f"   4. Bu script'teki DATASETS listesindeki bilgileri güncelle")
        print(f"   5. Dataset'in public olduğundan emin ol")
        return False


def verify_downloads():
    """İndirilen dataset'lerin yapısını kontrol et."""
    print(f"\n{'='*60}")
    print("📋 İndirme Doğrulama Raporu")
    print(f"{'='*60}")

    for dataset_info in DATASETS:
        name = dataset_info["name"]
        target = RAW_DIR / dataset_info["target_dir"]

        if not target.exists():
            print(f"❌ {name}: Klasör bulunamadı")
            continue

        # Alt klasörleri say
        splits = {}
        for split in ["train", "valid", "test"]:
            img_dir = target / split / "images"
            if img_dir.exists():
                count = len(list(img_dir.glob("*")))
                splits[split] = count
            else:
                splits[split] = 0

        total = sum(splits.values())
        print(f"\n📁 {name} ({total} görsel toplam)")
        for split, count in splits.items():
            bar = "█" * min(count // 50, 30)
            print(f"   {split:6s}: {count:5d} {bar}")

        # data.yaml sınıflarını göster
        yaml_path = target / "data.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            names = config.get("names", {})
            print(f"   Sınıflar: {names}")


def main():
    """Ana indirme fonksiyonu."""
    print("🔌 EV Şarj İstasyonu — Dataset İndirici")
    print("=" * 60)

    # API key kontrolü
    check_api_key()

    # Çıktı dizinini oluştur
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 İndirme dizini: {RAW_DIR}")

    # Dataset'leri indir
    results = {}
    for dataset_info in DATASETS:
        success = download_dataset(dataset_info)
        results[dataset_info["name"]] = success

    # Sonuç özeti
    print(f"\n{'='*60}")
    print("📊 İndirme Özeti")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✅ Başarılı" if success else "❌ Başarısız"
        print(f"  {status}  {name}")

    # Doğrulama
    verify_downloads()

    failed = [name for name, success in results.items() if not success]
    if failed:
        print(f"\n⚠️  {len(failed)} dataset indirilemedi: {', '.join(failed)}")
        print("Yukarıdaki hata mesajlarını kontrol edip workspace/project bilgilerini düzelt.")
        print("Ardından script'i tekrar çalıştır — başarılı olanlar atlanacak.")
    else:
        print(f"\n✅ Tüm dataset'ler başarıyla indirildi!")
        print(f"Sonraki adım: python scripts/merge_datasets.py")


if __name__ == "__main__":
    main()
