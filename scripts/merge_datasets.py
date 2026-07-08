"""
Dataset Birleştirme Script'i
=============================
İndirilen Roboflow dataset'lerini birleşik bir YOLOv8 dataset'ine dönüştürür.

İşlemler:
  1. Her dataset'in data.yaml'ını okur
  2. Sınıf eşleştirme tablosunu uygular (farklı isimleri birleştirir)
  3. Gereksiz sınıfları filtreler (Person, fire, smoke vb.)
  4. Dosya adı çakışmalarını prefix ile çözer
  5. Train/Val/Test split yapar
  6. Birleşik data.yaml oluşturur

Kullanım:
    python scripts/merge_datasets.py

Giriş:  datasets/raw/<dataset_adı>/
Çıkış:  datasets/merged/train|valid|test/images|labels/
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
MERGED_DIR = PROJECT_ROOT / "datasets" / "merged"

# Rastgelelik sabitlenmesi (tekrarlanabilirlik)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# BİRLEŞİK SINIF HARİTASI
# ============================================================
# Hedef: 3 sınıf
#   0 = vehicle
#   1 = charging_cable
#   2 = charging_socket
# ============================================================

UNIFIED_CLASSES = {
    0: "vehicle",
    1: "charging_cable",
    2: "charging_socket",
}

# Her dataset için: orijinal sınıf adı → birleşik sınıf ID
# None = bu sınıf filtrelenir (kullanılmaz)
CLASS_MAPPING = {
    "evchargeaero": {
        "car": 0,                  # → vehicle
        "charging cable": 1,       # → charging_cable
        "charging socket": 2,      # → charging_socket
        # Olası varyasyonlar
        "charging_cable": 1,
        "charging_socket": 2,
    },
    "ev_charging_10k": {
        "Car": 0,                  # → vehicle
        "car": 0,
        "Person": None,            # ❌ Filtrelenir
        "person": None,
        "Socket": 2,               # → charging_socket
        "socket": 2,
    },
    "ev_charging_471": {
        "CCS1AC": 1,               # → charging_cable (konnektör tipi)
        "CCS1DC": 1,               # → charging_cable
        "CCS2DC": 1,               # → charging_cable
        "Charging": 1,             # → charging_cable
        "charging": 1,
        "Socket": 2,               # → charging_socket
        "socket": 2,
    },
    "parking_lot": {
        # Parking lot dataset'inde sınıflar genellikle:
        # 0 = empty (boş park yeri)
        # 1 = occupied (dolu park yeri)
        # Sadece "occupied" (dolu) örneklerini vehicle olarak kullanacağız
        "0": None,                 # ❌ Boş park yeri, filtrelenir
        "1": 0,                    # → vehicle
        "occupied": 0,
        "empty": None,
        # Eğer sınıf isimleri sayısal ise
        0: None,                   # ❌ Boş
        1: 0,                      # → vehicle
    },
}

# Train/Val/Test oranları
SPLIT_RATIOS = {
    "train": 0.70,
    "valid": 0.20,
    "test": 0.10,
}


def load_dataset_config(dataset_dir: Path) -> dict:
    """Dataset'in data.yaml dosyasını oku."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"  ⚠️  data.yaml bulunamadı: {yaml_path}")
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_class_name_by_id(config: dict, class_id: int) -> str:
    """data.yaml'daki sınıf ID'sinden sınıf adını döndür."""
    names = config.get("names", {})

    # names liste mi dict mi kontrol et
    if isinstance(names, list):
        if class_id < len(names):
            return names[class_id]
        return str(class_id)
    elif isinstance(names, dict):
        return names.get(class_id, str(class_id))
    return str(class_id)


def remap_label_file(
    src_label_path: Path,
    dst_label_path: Path,
    dataset_name: str,
    config: dict,
) -> dict:
    """
    Tek bir label dosyasını oku, sınıf ID'lerini dönüştür, kaydet.

    Returns:
        dict: Her birleşik sınıftan kaç adet etiket olduğu
    """
    mapping = CLASS_MAPPING.get(dataset_name, {})
    stats = Counter()
    new_lines = []

    if not src_label_path.exists():
        return stats

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue  # Geçersiz satır

        orig_class_id = int(parts[0])
        orig_class_name = get_class_name_by_id(config, orig_class_id)

        # Eşleştirme tablosundan yeni sınıf ID'sini bul
        new_class_id = None

        # Önce sınıf adıyla dene
        if orig_class_name in mapping:
            new_class_id = mapping[orig_class_name]
        # Sonra sınıf ID'siyle dene
        elif orig_class_id in mapping:
            new_class_id = mapping[orig_class_id]
        # Son olarak string ID ile dene
        elif str(orig_class_id) in mapping:
            new_class_id = mapping[str(orig_class_id)]

        # None = filtrelenir
        if new_class_id is None:
            continue

        # Yeni satır oluştur
        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))
        stats[new_class_id] += 1

    # Eğer geçerli etiket varsa kaydet
    if new_lines:
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

    return stats


def process_dataset(dataset_name: str) -> list:
    """
    Tek bir dataset'i işle: label'ları dönüştür, görsel + label çiftlerini listele.

    Returns:
        list of tuples: [(src_img, src_lbl, dst_img_name, dst_lbl_name), ...]
    """
    dataset_dir = RAW_DIR / dataset_name
    if not dataset_dir.exists():
        print(f"  ❌ Dataset dizini bulunamadı: {dataset_dir}")
        return []

    config = load_dataset_config(dataset_dir)
    if config is None:
        # data.yaml olmadan devam etmeyi dene
        config = {"names": {}}

    print(f"\n  📁 {dataset_name}")
    print(f"     Dizin: {dataset_dir}")
    if config.get("names"):
        print(f"     Orijinal sınıflar: {config['names']}")

    pairs = []
    total_stats = Counter()

    # Train, valid, test split'lerini tara
    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*"))
        for img_path in images:
            # Desteklenen formatlar
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            # Karşılık gelen label dosyası
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            # Yeni dosya adları (prefix ile çakışma önleme)
            dst_img_name = f"{dataset_name}_{img_path.name}"
            dst_lbl_name = f"{dataset_name}_{img_path.stem}.txt"

            # Label dosyasını geçici olarak dönüştür ve kontrol et
            if lbl_path.exists():
                # Label'daki sınıfları oku
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                has_valid_label = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_class_id = int(parts[0])
                        orig_class_name = get_class_name_by_id(config, orig_class_id)
                        mapping = CLASS_MAPPING.get(dataset_name, {})

                        new_id = None
                        if orig_class_name in mapping:
                            new_id = mapping[orig_class_name]
                        elif orig_class_id in mapping:
                            new_id = mapping[orig_class_id]
                        elif str(orig_class_id) in mapping:
                            new_id = mapping[str(orig_class_id)]

                        if new_id is not None:
                            has_valid_label = True
                            total_stats[new_id] += 1

                if has_valid_label:
                    pairs.append((img_path, lbl_path, dst_img_name, dst_lbl_name, config))

    print(f"     Geçerli görsel-label çifti: {len(pairs)}")
    for cls_id, count in sorted(total_stats.items()):
        cls_name = UNIFIED_CLASSES.get(cls_id, f"unknown_{cls_id}")
        print(f"       {cls_name}: {count} etiket")

    return pairs


def split_and_copy(all_pairs: list):
    """
    Tüm çiftleri karıştır, train/val/test'e böl, kopyala.
    """
    # Karıştır
    random.shuffle(all_pairs)

    total = len(all_pairs)
    train_end = int(total * SPLIT_RATIOS["train"])
    valid_end = train_end + int(total * SPLIT_RATIOS["valid"])

    splits = {
        "train": all_pairs[:train_end],
        "valid": all_pairs[train_end:valid_end],
        "test": all_pairs[valid_end:],
    }

    print(f"\n📊 Split Dağılımı:")
    for split_name, split_pairs in splits.items():
        print(f"   {split_name}: {len(split_pairs)} görsel ({len(split_pairs)/total*100:.1f}%)")

    # Hedef dizinleri oluştur
    for split_name in splits:
        (MERGED_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # Kopyala
    overall_stats = Counter()

    for split_name, split_pairs in splits.items():
        print(f"\n   📂 {split_name} kopyalanıyor...")
        split_stats = Counter()

        for img_path, lbl_path, dst_img_name, dst_lbl_name, config in tqdm(split_pairs, desc=f"   {split_name}"):
            # Görseli kopyala
            dst_img = MERGED_DIR / split_name / "images" / dst_img_name
            shutil.copy2(img_path, dst_img)

            # Label'ı dönüştürüp kaydet
            dst_lbl = MERGED_DIR / split_name / "labels" / dst_lbl_name
            dataset_name = dst_img_name.split("_")[0]
            # Prefix'ten dataset adını çıkar
            for ds_name in CLASS_MAPPING:
                if dst_img_name.startswith(ds_name + "_"):
                    dataset_name = ds_name
                    break

            stats = remap_label_file(lbl_path, dst_lbl, dataset_name, config)
            split_stats.update(stats)

        overall_stats.update(split_stats)

        print(f"   {split_name} etiket dağılımı:")
        for cls_id, count in sorted(split_stats.items()):
            cls_name = UNIFIED_CLASSES.get(cls_id, f"unknown_{cls_id}")
            print(f"     {cls_name}: {count}")

    return overall_stats


def create_data_yaml():
    """Birleşik dataset için data.yaml oluştur."""
    config = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }

    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n📄 data.yaml oluşturuldu: {yaml_path}")
    print(f"   İçerik:")
    with open(yaml_path, "r", encoding="utf-8") as f:
        print(f"   {f.read()}")

    # Ayrıca proje kökündeki datasets/ altına da kopyala
    alt_yaml = PROJECT_ROOT / "datasets" / "data.yaml"
    shutil.copy2(yaml_path, alt_yaml)
    print(f"   Kopyalandı: {alt_yaml}")


def verify_merged_dataset():
    """Birleşik dataset'in bütünlüğünü doğrula."""
    print(f"\n{'='*60}")
    print("🔍 Doğrulama Raporu")
    print(f"{'='*60}")

    issues = []

    for split in ["train", "valid", "test"]:
        img_dir = MERGED_DIR / split / "images"
        lbl_dir = MERGED_DIR / split / "labels"

        if not img_dir.exists():
            issues.append(f"{split}/images dizini bulunamadı")
            continue

        images = set(p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        labels = set(p.stem for p in lbl_dir.glob("*.txt"))

        # Eşleşme kontrolü
        missing_labels = images - labels
        orphan_labels = labels - images

        print(f"\n   {split}:")
        print(f"     Görseller: {len(images)}")
        print(f"     Label'lar: {len(labels)}")

        if missing_labels:
            print(f"     ⚠️  Label'sız görseller: {len(missing_labels)}")
            if len(missing_labels) <= 5:
                for name in sorted(missing_labels):
                    print(f"        - {name}")
            issues.append(f"{split}: {len(missing_labels)} görsel label'sız")

        if orphan_labels:
            print(f"     ⚠️  Görselsiz label'lar: {len(orphan_labels)}")
            issues.append(f"{split}: {len(orphan_labels)} yetim label")

        # Rastgele bir label'ı kontrol et
        if labels:
            sample_lbl = lbl_dir / f"{sorted(labels)[0]}.txt"
            with open(sample_lbl, "r") as f:
                first_line = f.readline().strip()
            parts = first_line.split()
            if parts:
                cls_id = int(parts[0])
                cls_name = UNIFIED_CLASSES.get(cls_id, "BİLİNMEYEN")
                print(f"     Örnek label: sınıf={cls_id} ({cls_name}), bbox={parts[1:5]}")

    if issues:
        print(f"\n⚠️  {len(issues)} sorun tespit edildi:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Dataset doğrulama başarılı — sorun yok!")

    return len(issues) == 0


def main():
    """Ana birleştirme fonksiyonu."""
    print("🔌 EV Şarj İstasyonu — Dataset Birleştirici")
    print("=" * 60)

    # Ham dataset'leri kontrol et
    if not RAW_DIR.exists():
        print(f"❌ Ham dataset dizini bulunamadı: {RAW_DIR}")
        print("Önce download_datasets.py çalıştırın.")
        sys.exit(1)

    available = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"📂 Bulunan dataset'ler: {available}")

    if not available:
        print("❌ Hiç dataset bulunamadı! Önce download_datasets.py çalıştırın.")
        sys.exit(1)

    # Eski birleşik dataset'i temizle
    if MERGED_DIR.exists():
        print(f"\n🧹 Eski birleşik dataset temizleniyor: {MERGED_DIR}")
        shutil.rmtree(MERGED_DIR)

    # Her dataset'i işle
    all_pairs = []
    for dataset_name in available:
        if dataset_name in CLASS_MAPPING:
            pairs = process_dataset(dataset_name)
            all_pairs.extend(pairs)
        else:
            print(f"\n  ⏭️  {dataset_name}: Sınıf eşleştirmesi tanımlı değil, atlanıyor")

    if not all_pairs:
        print("\n❌ Hiç geçerli görsel-label çifti bulunamadı!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📊 Toplam {len(all_pairs)} görsel-label çifti bulundu")
    print(f"{'='*60}")

    # Böl ve kopyala
    overall_stats = split_and_copy(all_pairs)

    # data.yaml oluştur
    create_data_yaml()

    # Doğrula
    success = verify_merged_dataset()

    # Sonuç özeti
    print(f"\n{'='*60}")
    print("📊 Birleştirme Özeti")
    print(f"{'='*60}")
    print(f"   Toplam görseller: {len(all_pairs)}")
    print(f"   Toplam etiketler: {sum(overall_stats.values())}")
    print(f"   Sınıf dağılımı:")
    for cls_id, count in sorted(overall_stats.items()):
        cls_name = UNIFIED_CLASSES.get(cls_id, f"unknown_{cls_id}")
        pct = count / sum(overall_stats.values()) * 100
        bar = "█" * int(pct / 2)
        print(f"     {cls_name:20s}: {count:6d} ({pct:5.1f}%) {bar}")

    if success:
        print(f"\n✅ Birleştirme tamamlandı!")
        print(f"   Dataset: {MERGED_DIR}")
        print(f"   Config:  {MERGED_DIR / 'data.yaml'}")
        print(f"\n   Sonraki adım: python scripts/train.py")
    else:
        print(f"\n⚠️  Birleştirme tamamlandı ama bazı sorunlar var. Yukarıdaki uyarıları kontrol et.")


if __name__ == "__main__":
    main()
