"""
Model Değerlendirme Script'i
==============================
Eğitilmiş YOLOv8 modelinin performansını test seti üzerinde değerlendirir.

Metrikler:
  - mAP@0.5, mAP@0.5:0.95
  - Precision, Recall, F1-Score (per-class ve ortalama)
  - Confusion Matrix
  - Inference time
  - Hatalı tahmin örnekleri

Kullanım:
    python scripts/evaluate.py
    python scripts/evaluate.py --model results/ev_charging/weights/best.pt
    python scripts/evaluate.py --save-errors
"""

import argparse
import time
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalışsın
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "merged" / "data.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL = RESULTS_DIR / "ev_charging" / "weights" / "best.pt"

CLASS_NAMES = {0: "vehicle", 1: "charging_cable", 2: "charging_socket"}


def parse_args():
    parser = argparse.ArgumentParser(description="EV Şarj İstasyonu — Model Değerlendirme")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model yolu")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Dataset YAML yolu")
    parser.add_argument("--imgsz", type=int, default=640, help="Görsel boyutu")
    parser.add_argument("--batch", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    parser.add_argument("--save-errors", action="store_true", help="Hatalı tahminleri kaydet")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    return parser.parse_args()


def evaluate_model(model, args):
    """YOLOv8 built-in değerlendirmesi."""
    print("\n📊 Model Değerlendirmesi (Test Set)")
    print("=" * 50)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="test",
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        verbose=True,
    )

    return results


def print_metrics(results):
    """Metrikleri güzel formatta yazdır."""
    print(f"\n{'='*60}")
    print("📈 Performans Metrikleri")
    print(f"{'='*60}")

    # Genel metrikler
    box = results.box
    print(f"\n  Genel Metrikler:")
    print(f"  {'─'*40}")
    print(f"  mAP@0.5:       {box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {box.map:.4f}")
    print(f"  Precision:      {box.mp:.4f}")
    print(f"  Recall:         {box.mr:.4f}")

    # F1 hesapla
    if box.mp + box.mr > 0:
        f1 = 2 * (box.mp * box.mr) / (box.mp + box.mr)
    else:
        f1 = 0.0
    print(f"  F1-Score:       {f1:.4f}")

    # Per-class metrikler
    print(f"\n  Sınıf Bazlı Metrikler:")
    print(f"  {'─'*55}")
    print(f"  {'Sınıf':<20s} {'Precision':>10s} {'Recall':>10s} {'mAP@0.5':>10s} {'mAP@.5:.95':>10s}")
    print(f"  {'─'*55}")

    # Per-class AP değerleri
    for i, cls_name in CLASS_NAMES.items():
        try:
            ap50 = box.ap50[i] if i < len(box.ap50) else 0.0
            ap = box.ap[i] if i < len(box.ap) else 0.0
            p = box.p[i] if i < len(box.p) else 0.0
            r = box.r[i] if i < len(box.r) else 0.0
            print(f"  {cls_name:<20s} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {ap:>10.4f}")
        except (IndexError, AttributeError):
            print(f"  {cls_name:<20s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")

    # Hedef kontrolü
    print(f"\n  🎯 Hedef Kontrolü:")
    print(f"  {'─'*40}")

    targets = {
        "mAP@0.5 ≥ 0.85": box.map50 >= 0.85,
        "Precision ≥ 0.90": box.mp >= 0.90,
        "Recall ≥ 0.85": box.mr >= 0.85,
        "F1 ≥ 0.87": f1 >= 0.87,
    }

    for target_name, achieved in targets.items():
        status = "✅" if achieved else "❌"
        print(f"  {status} {target_name}")

    return f1


def benchmark_inference(model, args, n_runs=100):
    """Inference hızını ölçer."""
    print(f"\n⏱️  Inference Benchmark ({n_runs} çalıştırma)")
    print("=" * 50)

    import torch

    # Dummy input
    dummy = torch.rand(1, 3, args.imgsz, args.imgsz)
    if args.device != "cpu" and torch.cuda.is_available():
        dummy = dummy.to(f"cuda:{args.device}")

    # Warm-up
    for _ in range(10):
        model.predict(dummy, verbose=False)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)
    print(f"  Ortalama: {times.mean():.1f} ms")
    print(f"  Median:   {np.median(times):.1f} ms")
    print(f"  Min:      {times.min():.1f} ms")
    print(f"  Max:      {times.max():.1f} ms")
    print(f"  Std:      {times.std():.1f} ms")
    print(f"  FPS:      {1000 / times.mean():.1f}")

    target_ms = 30
    status = "✅" if times.mean() <= target_ms else "❌"
    print(f"\n  {status} Hedef: ≤ {target_ms}ms (Gerçek: {times.mean():.1f}ms)")

    return times.mean()


def save_evaluation_report(results, f1, avg_inference_ms, args):
    """Değerlendirme raporunu JSON olarak kaydet."""
    report_dir = RESULTS_DIR / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)

    box = results.box
    report = {
        "model": args.model,
        "dataset": args.data,
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "metrics": {
            "mAP_50": float(box.map50),
            "mAP_50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1_score": float(f1),
        },
        "per_class": {},
        "inference_ms": float(avg_inference_ms),
    }

    for i, cls_name in CLASS_NAMES.items():
        try:
            report["per_class"][cls_name] = {
                "precision": float(box.p[i]) if i < len(box.p) else None,
                "recall": float(box.r[i]) if i < len(box.r) else None,
                "ap50": float(box.ap50[i]) if i < len(box.ap50) else None,
                "ap": float(box.ap[i]) if i < len(box.ap) else None,
            }
        except (IndexError, AttributeError):
            report["per_class"][cls_name] = None

    report_path = report_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Rapor kaydedildi: {report_path}")
    return report_path


def create_summary_plot(results, f1, avg_inference_ms):
    """Özet performans grafiği oluştur."""
    report_dir = RESULTS_DIR / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)

    box = results.box

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EV Şarj İstasyonu — Model Performans Özeti", fontsize=14, fontweight="bold")

    # 1. Genel metrikler bar chart
    metrics = {
        "mAP@0.5": box.map50,
        "mAP@.5:.95": box.map,
        "Precision": box.mp,
        "Recall": box.mr,
        "F1": f1,
    }
    colors = ["#2196F3" if v >= 0.85 else "#FF5722" for v in metrics.values()]
    bars = axes[0].bar(metrics.keys(), metrics.values(), color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Genel Metrikler")
    axes[0].axhline(y=0.85, color="green", linestyle="--", alpha=0.5, label="Hedef (0.85)")
    axes[0].legend()
    for bar, val in zip(bars, metrics.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # 2. Per-class mAP
    class_names = []
    class_ap50 = []
    for i, cls_name in CLASS_NAMES.items():
        try:
            class_names.append(cls_name)
            class_ap50.append(float(box.ap50[i]) if i < len(box.ap50) else 0.0)
        except (IndexError, AttributeError):
            class_names.append(cls_name)
            class_ap50.append(0.0)

    colors2 = ["#4CAF50" if v >= 0.85 else "#FFC107" if v >= 0.7 else "#F44336" for v in class_ap50]
    bars2 = axes[1].barh(class_names, class_ap50, color=colors2, edgecolor="white", linewidth=0.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Sınıf Bazlı mAP@0.5")
    axes[1].axvline(x=0.85, color="green", linestyle="--", alpha=0.5)
    for bar, val in zip(bars2, class_ap50):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", ha="left", va="center", fontsize=10)

    # 3. Inference speed
    axes[2].bar(["Inference Time"], [avg_inference_ms], color="#9C27B0", width=0.4)
    axes[2].axhline(y=30, color="green", linestyle="--", alpha=0.5, label="Hedef (30ms)")
    axes[2].set_ylabel("ms")
    axes[2].set_title("Inference Hızı")
    axes[2].legend()
    axes[2].text(0, avg_inference_ms + 1, f"{avg_inference_ms:.1f}ms\n({1000/avg_inference_ms:.0f} FPS)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plot_path = report_dir / "performance_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Özet grafik: {plot_path}")


def main():
    args = parse_args()

    print("🔌 EV Şarj İstasyonu — Model Değerlendirme")
    print("=" * 50)

    # Model kontrolü
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model bulunamadı: {model_path}")
        print("   Önce train.py ile model eğitin.")

        # Mevcut modelleri listele
        existing = list(RESULTS_DIR.rglob("*.pt"))
        if existing:
            print(f"\n   Mevcut modeller:")
            for p in existing:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"     {p} ({size_mb:.1f} MB)")
        sys.exit(1)

    print(f"✅ Model: {model_path}")

    # Modeli yükle
    model = YOLO(str(model_path))

    # Değerlendir
    results = evaluate_model(model, args)

    # Metrikleri yazdır
    f1 = print_metrics(results)

    # Inference benchmark
    avg_ms = benchmark_inference(model, args)

    # Rapor kaydet
    save_evaluation_report(results, f1, avg_ms, args)

    # Özet grafik
    try:
        create_summary_plot(results, f1, avg_ms)
    except Exception as e:
        print(f"⚠️  Grafik oluşturulamadı: {e}")

    print(f"\n{'='*50}")
    print("✅ Değerlendirme tamamlandı!")
    print(f"   Sonuçlar: {RESULTS_DIR / 'evaluation'}")


if __name__ == "__main__":
    main()
