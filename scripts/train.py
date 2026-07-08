"""
YOLOv8 Fine-Tuning Script'i
============================
Birleştirilmiş EV şarj istasyonu dataset'i üzerinde YOLOv8 modelini eğitir.

Kullanım:
    python scripts/train.py
    python scripts/train.py --model yolov8s.pt --epochs 100 --batch 32
    python scripts/train.py --resume  (kaldığı yerden devam)

RTX 4080 Mobile 12GB VRAM için optimize edilmiştir.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "merged" / "data.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args():
    """Komut satırı argümanlarını parse et."""
    parser = argparse.ArgumentParser(
        description="EV Şarj İstasyonu — YOLOv8 Eğitim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python scripts/train.py                          # Varsayılan ayarlar
  python scripts/train.py --model yolov8s.pt       # Small model
  python scripts/train.py --epochs 200 --batch 8   # Uzun eğitim, düşük batch
  python scripts/train.py --resume                 # Son eğitimi devam ettir
        """,
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="Base model (yolov8n/s/m/l/x.pt) (varsayılan: yolov8m.pt)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Son eğitimi kaldığı yerden devam ettir"
    )

    # Eğitim parametreleri
    parser.add_argument("--epochs", type=int, default=150, help="Epoch sayısı (varsayılan: 150)")
    parser.add_argument("--batch", type=int, default=16, help="Batch boyutu (varsayılan: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Görsel boyutu (varsayılan: 640)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (varsayılan: 30)")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (varsayılan: AdamW)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Başlangıç learning rate (varsayılan: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR çarpanı (varsayılan: 0.01)")
    parser.add_argument("--warmup_epochs", type=float, default=5.0, help="Warmup epoch sayısı (varsayılan: 5)")

    # Augmentation
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation olasılığı")
    parser.add_argument("--mixup", type=float, default=0.1, help="MixUp augmentation olasılığı")
    parser.add_argument("--degrees", type=float, default=15.0, help="Rotation açısı (derece)")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Yatay flip olasılığı")
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV-Hue augmentation")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV-Saturation augmentation")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV-Value augmentation")

    # Donanım
    parser.add_argument("--device", type=str, default="0", help="GPU device (varsayılan: 0)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker sayısı")

    # Çıktı
    parser.add_argument("--name", type=str, default="ev_charging", help="Deneme adı")

    return parser.parse_args()


def validate_setup(args):
    """Eğitim öncesi kontroller."""
    print("🔍 Ön Kontroller")
    print("=" * 50)

    # Dataset kontrolü
    if not DATA_YAML.exists():
        print(f"❌ data.yaml bulunamadı: {DATA_YAML}")
        print("   Önce merge_datasets.py çalıştırın.")
        sys.exit(1)

    import yaml
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"✅ Dataset config: {DATA_YAML}")
    print(f"   Sınıf sayısı: {config['nc']}")
    print(f"   Sınıflar: {config['names']}")

    # Görselleri say
    train_imgs = Path(config["path"]) / "train" / "images"
    val_imgs = Path(config["path"]) / "valid" / "images"

    if train_imgs.exists():
        train_count = len(list(train_imgs.glob("*")))
        print(f"   Train görselleri: {train_count}")
    else:
        print(f"❌ Train dizini bulunamadı: {train_imgs}")
        sys.exit(1)

    if val_imgs.exists():
        val_count = len(list(val_imgs.glob("*")))
        print(f"   Valid görselleri: {val_count}")

    # GPU kontrolü
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Batch boyutu önerisi
        if gpu_mem < 8:
            recommended_batch = 8
        elif gpu_mem < 12:
            recommended_batch = 16
        else:
            recommended_batch = 32

        if args.batch > recommended_batch:
            print(f"   ⚠️  Batch={args.batch} VRAM için yüksek olabilir. Önerilen: {recommended_batch}")
    else:
        print("⚠️  GPU bulunamadı! CPU ile eğitim çok yavaş olacaktır.")
        args.device = "cpu"

    # Model kontrolü
    print(f"✅ Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Patience: {args.patience}")
    print(f"   Optimizer: {args.optimizer}")
    print(f"   LR: {args.lr0} → {args.lr0 * args.lrf}")

    return config


def train(args):
    """Ana eğitim fonksiyonu."""
    print(f"\n🚀 Eğitim Başlatılıyor")
    print("=" * 50)

    # Modeli yükle
    if args.resume:
        # Son eğitimin checkpoint'inden devam
        last_pt = RESULTS_DIR / args.name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"📂 Devam ediliyor: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"❌ Checkpoint bulunamadı: {last_pt}")
            print("   İlk eğitimi başlatıyorum...")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    # Eğitimi başlat
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        # Augmentation
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        flipud=0.0,      # Dikey flip kapalı (araba ters olmaz)
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        # Scale augmentation (küçük nesneler için önemli)
        scale=0.5,
        translate=0.1,
        # Donanım
        device=args.device,
        workers=args.workers,
        # Çıktı
        project=str(RESULTS_DIR),
        name=args.name,
        exist_ok=True,    # Aynı adla üzerine yaz
        verbose=True,
        # Kaydetme
        save=True,
        save_period=10,   # Her 10 epoch'ta checkpoint
        plots=True,       # Eğitim grafikleri
        resume=args.resume, # <--- Ultralytics'in kaldığı epoch'u anlaması için GEREKLİ
    )

    return results


def post_training_summary(args):
    """Eğitim sonrası özet."""
    print(f"\n{'='*50}")
    print("📊 Eğitim Tamamlandı!")
    print(f"{'='*50}")

    results_dir = RESULTS_DIR / args.name
    best_model = results_dir / "weights" / "best.pt"
    last_model = results_dir / "weights" / "last.pt"

    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"✅ En iyi model: {best_model} ({size_mb:.1f} MB)")
    if last_model.exists():
        print(f"✅ Son model: {last_model}")

    # Eğitim grafikleri
    plots = list(results_dir.glob("*.png"))
    if plots:
        print(f"\n📈 Eğitim grafikleri ({len(plots)} dosya):")
        for plot in sorted(plots):
            print(f"   {plot.name}")

    print(f"\n🎯 Sonraki adımlar:")
    print(f"   1. Değerlendirme: python scripts/evaluate.py")
    print(f"   2. Pipeline testi: python scripts/pipeline.py --source test_image.jpg")
    print(f"   3. Demo: python app/demo.py")


def main():
    args = parse_args()

    print("🔌 EV Şarj İstasyonu — YOLOv8 Eğitim")
    print("=" * 50)

    # Kontroller
    validate_setup(args)

    # Eğitim
    results = train(args)

    # Özet
    post_training_summary(args)


if __name__ == "__main__":
    main()
