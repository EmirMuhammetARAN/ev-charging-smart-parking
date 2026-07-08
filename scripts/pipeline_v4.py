import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "ev_charging_v4" / "weights" / "best.pt"

class EVChargingPipelineV4:
    """
    EV Şarj İstasyonu Akıllı Yönetim Pipeline'ı (V4)
    YOLOv8 8-Sınıflı Model (Gece, Sis ve Anomali destekli)
    """

    def __init__(self, model_path: str, confidence: float = 0.5):
        print(f"🔄 Model yükleniyor: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence

        # V4 Modelindeki 8 Sınıfımız
        self.class_names = {
            0: "Fis_Bosta",
            1: "Fis_Takili",
            2: "Fis_yerde",
            3: "car_charging",
            4: "car_parked",
            5: "station_bosta",
            6: "station_park",
            7: "station_sarj"
        }
        
        # UI İçin Renk Kodları (BGR)
        self.colors = {
            "Fis_Bosta": (200, 200, 200),     # Gri
            "Fis_Takili": (0, 255, 255),      # Sarı
            "Fis_yerde": (0, 0, 255),         # Kırmızı (Anomali)
            "car_charging": (255, 150, 0),    # Turkuaz/Mavi
            "car_parked": (0, 150, 255),      # Turuncu
            "station_bosta": (0, 255, 0),     # Yeşil
            "station_park": (0, 100, 255),    # Koyu Turuncu (İhlal)
            "station_sarj": (255, 50, 50)     # Koyu Mavi
        }

        print("✅ V4 Akıllı Pipeline hazır!")

    def process_frame(self, frame: np.ndarray) -> dict:
        """Tek bir frame'i YOLOv4 modelimiz ile analiz et."""
        frame_start = time.perf_counter()
        inference_start = time.perf_counter()
        results = self.model(frame, conf=self.confidence, verbose=False)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        
        detections = []
        counts = {name: 0 for name in self.class_names.values()}
        alerts = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [int(v) for v in xyxy]
                })
                
                if cls_name in counts:
                    counts[cls_name] += 1

        # --- AKILLI ALARM MANTIĞI ---
        # 1. Kritik Anomali: Fiş Yerde Sürünüyor!
        if counts["Fis_yerde"] > 0:
            alerts.append("Kritik Anomali: Fis Yerde!")
            
        # 2. Park İhlali: Araç park etmiş ama şarj olmuyor!
        if counts["station_park"] > 0:
            alerts.append(f"Park Ihlali: {counts['station_park']} arac bosuna bekliyor!")

        # Annotated frame oluştur
        total_ms = (time.perf_counter() - frame_start) * 1000
        fps = 1000 / total_ms if total_ms > 0 else 0
        performance = {
            "inference_ms": inference_ms,
            "total_ms": total_ms,
            "fps": fps,
        }

        annotated = self._annotate_frame(frame, detections, counts, alerts, performance)

        return {
            "detections": detections,
            "counts": counts,
            "alerts": alerts,
            "performance": performance,
            "annotated_frame": annotated,
        }

    def _annotate_frame(self, frame: np.ndarray, detections: list, counts: dict, alerts: list, performance: dict) -> np.ndarray:
        """Frame üzerine Bounding Box'ları ve UI Paneli Çiz."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Bounding box'ları çiz
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = self.colors.get(cls_name, (128, 128, 128))

            # Kalınlık ve görsel ayarı
            thickness = 3 if cls_name in ["Fis_yerde", "station_park"] else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Etiket arka planı ve yazı
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # --- DURUM PANELİ ÇİZİMİ ---
        panel_w = 380
        panel_h = 240 + (len(alerts) * 30)
        
        # Yarı saydam arka plan
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        y_offset = 40
        cv2.putText(annotated, "TUBITAK Akilli Sarj Yonetimi V4", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.line(annotated, (20, y_offset + 10), (panel_w - 10, y_offset + 10), (255, 255, 255), 1)
        y_offset += 35

        # İstatistikler
        stats = [
            (f"Bos Istasyonlar: {counts['station_bosta']}", (0, 255, 0)),
            (f"Aktif Sarj (Istasyon): {counts['station_sarj']}", (255, 255, 0)),
            (f"Park Ihlali (Istasyon): {counts['station_park']}", (0, 150, 255)),
            (f"Yerdeki Fis: {counts['Fis_yerde']}", (0, 0, 255)),
            (f"Inference: {performance['inference_ms']:.1f} ms | FPS: {performance['fps']:.1f}", (255, 255, 255))
        ]

        for text, color in stats:
            cv2.putText(annotated, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y_offset += 30

        # Alarmlar
        if alerts:
            y_offset += 10
            cv2.putText(annotated, "--- AKTIF ALARMLAR ---", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            y_offset += 30
            for alert in alerts:
                # Arka planla kırmızı dikkat çekici uyarı
                cv2.rectangle(annotated, (15, y_offset - 20), (panel_w, y_offset + 5), (0, 0, 200), -1)
                cv2.putText(annotated, alert, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 30

        return annotated

    def process_video(self, source, save_path: str = None, show: bool = True):
        """Video veya webcam işle."""
        if isinstance(source, int) or source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print(f"📸 Kamera açıldı: {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"🎥 Video açıldı: {source}")

        if not cap.isOpened():
            raise ValueError(f"Kaynak açılamadı: {source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_count = 0
        total_inference_ms = 0
        total_processing_ms = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                result = self.process_frame(frame)
                total_inference_ms += result["performance"]["inference_ms"]
                total_processing_ms += result["performance"]["total_ms"]
                annotated = result["annotated_frame"]

                if writer:
                    writer.write(annotated)

                if show:
                    # Ekrana sığması için yeniden boyutlandır (Opsiyonel)
                    disp_frame = cv2.resize(annotated, (1280, 720)) if width > 1280 else annotated
                    cv2.imshow("EV Sarj Istasyonu V4", disp_frame)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        break

        except KeyboardInterrupt:
            print("\n⏹️ Durduruldu (Ctrl+C)")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            if frame_count > 0:
                avg_inference_ms = total_inference_ms / frame_count
                avg_fps = 1000 / (total_processing_ms / frame_count) if total_processing_ms > 0 else 0
                print(f"\nOrtalama inference: {avg_inference_ms:.2f} ms | Ortalama FPS: {avg_fps:.2f}")
            print(f"\n✅ İşlem tamamlandı: {frame_count} frame işlendi")

    def process_image(self, image_path: str, save_path: str = None, show: bool = True):
        """Tek bir görseli işle."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Görsel okunamadı: {image_path}")
        
        result = self.process_frame(frame)
        perf = result["performance"]
        print(f"Inference: {perf['inference_ms']:.2f} ms | Toplam: {perf['total_ms']:.2f} ms | FPS: {perf['fps']:.2f}")
        
        if save_path:
            cv2.imwrite(save_path, result["annotated_frame"])
            
        if show:
            disp_frame = cv2.resize(result["annotated_frame"], (1280, 720)) if frame.shape[1] > 1280 else result["annotated_frame"]
            cv2.imshow("EV Sarj Istasyonu V4", disp_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="EV Şarj İstasyonu Akıllı Yönetim Pipeline V4")
    parser.add_argument("--source", type=str, required=True, help="Kaynak: görsel yolu, video yolu veya kamera index (0)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model yolu (.pt)")
    parser.add_argument("--conf", type=float, default=0.5, help="Minimum güven skoru")
    parser.add_argument("--save", type=str, default=None, help="Çıktı kaydedilecek dosya yolu (.mp4 veya .jpg)")
    parser.add_argument("--no-show", action="store_true", help="Pencere gösterme")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Model bulunamadı: {args.model}")
        sys.exit(1)

    pipeline = EVChargingPipelineV4(model_path=args.model, confidence=args.conf)
    source = args.source

    if source.isdigit():
        pipeline.process_video(source, save_path=args.save, show=not args.no_show)
    elif Path(source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
        pipeline.process_image(source, save_path=args.save, show=not args.no_show)
    else:
        pipeline.process_video(source, save_path=args.save, show=not args.no_show)

if __name__ == "__main__":
    main()
