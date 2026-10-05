import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Proje kok dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "ev_charging_v4" / "weights" / "best.pt"

class EVChargingPipelineV4:
    """
    EV Sarj Istasyonu Akilli Park Yonetimi ve Guvenlik Izleme Pipeline (V4)
    Minimalist, Ince Cizgili, Resim Alti Padding Panelli Profesyonel Versiyon
    """

    def __init__(self, model_path: str = str(DEFAULT_MODEL), confidence: float = 0.45):
        print(f"[*] Model yukleniyor: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Hizli GPU isitmasi (Warmup) - ilk karedeki gecikmeyi onler
        try:
            _ = self.model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        except Exception:
            pass

        # 8 Sinif Tanimlari
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
        
        # Turkce Kisa ve Net Etiketler
        self.display_names = {
            "Fis_Bosta": "Fis Bosta",
            "Fis_Takili": "Fis Takili",
            "Fis_yerde": "FIS YERDE!",
            "car_charging": "Sarj Olan",
            "car_parked": "Park Eden",
            "station_bosta": "Bos Slot",
            "station_park": "Ihlal Slot",
            "station_sarj": "Aktif Slot"
        }

        # Zarif & Sade Renk Paleti (BGR)
        self.colors = {
            "Fis_Bosta": (170, 150, 140),     # Metalik Gri
            "Fis_Takili": (235, 180, 50),     # Acik Mavi
            "Fis_yerde": (50, 50, 220),       # Kirmizi (Kritik)
            "car_charging": (220, 150, 20),   # Cyber Cyan
            "car_parked": (20, 140, 235),     # Kehribar Turuncusu
            "station_bosta": (110, 175, 30),  # Zumrut Yesili
            "station_park": (20, 90, 220),    # Koyu Turuncu
            "station_sarj": (190, 120, 10)    # Koyu Cyan
        }

        print("[+] V4 Minimalist Pipeline hazir!")

    def process_frame(self, frame: np.ndarray) -> dict:
        """Tek bir frame'i YOLOv8 modelimiz ile analiz et."""
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

        # --- ALARM MANTIGI ---
        if counts["Fis_yerde"] > 0:
            alerts.append(f"Guvenlik: {counts['Fis_yerde']} kablo yerde!")
            
        if counts["station_park"] > 0:
            alerts.append(f"Park Ihlali: {counts['station_park']} arac bekliyor")

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
        """
        Sade, temiz ve profesyonel cizim:
        - Bounding box cizgileri cok ince (1px), resmi bogmaz.
        - Durum paneli resmin uzerini asla kapatmaz; resmin ALTINA ferah 2 satirli bir padding bari olarak eklenir.
        - Tek gorsellerde 0.9 FPS gibi yaniltici sayilar yazilmaz, sadece model ve net inference suresi yazilir.
        """
        img_h, img_w = frame.shape[:2]
        img_draw = frame.copy()

        # 1. Bounding Box'lari Ciz (Incecik 1px cizgi)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = self.colors.get(cls_name, (160, 160, 160))
            disp_name = self.display_names.get(cls_name, cls_name)

            # Incecik 1px kutu
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # Minimalist, zarif kucuk etiket (fontScale 0.34, kalinlik 1)
            label = f"{disp_name} {int(conf * 100)}%"
            font_scale = 0.34
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            tag_h = th + 4
            tag_w = tw + 6
            tag_y1 = max(0, y1 - tag_h)
            tag_y2 = tag_y1 + tag_h
            tag_x1 = x1
            tag_x2 = min(img_w, x1 + tag_w)

            # Koyu mat etiket arka plani (hafif yarim saydam)
            overlay = img_draw[tag_y1:tag_y2, tag_x1:tag_x2].copy()
            bg_rect = np.full_like(overlay, (18, 22, 28))
            cv2.addWeighted(bg_rect, 0.82, overlay, 0.18, 0, overlay)
            img_draw[tag_y1:tag_y2, tag_x1:tag_x2] = overlay

            # Etiketin kenarina 1px renk cizgisi
            cv2.line(img_draw, (tag_x1, tag_y1), (tag_x1 + tag_w, tag_y1), color, 1, cv2.LINE_AA)

            # Metin
            cv2.putText(img_draw, label, (tag_x1 + 3, tag_y1 + th + 1), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

        # 2. Resmin ALTINA 62px Genisletilmis Footer Bari
        footer_h = 62
        final_h = img_h + footer_h
        final_canvas = np.zeros((final_h, img_w, 3), dtype=np.uint8)

        # Resmi ust kisma yerlestir (0% kapanma, goruntu tamamen gorunur)
        final_canvas[:img_h, :img_w] = img_draw

        # Alt bar arka plani (Derin mat antrasit)
        footer_bg = (20, 24, 30)  # BGR
        final_canvas[img_h:, :img_w] = footer_bg

        # Ust ince ayirici cizgi (1px slate border)
        cv2.line(final_canvas, (0, img_h), (img_w, img_h), (45, 55, 70), 1, cv2.LINE_AA)

        # --- SATIR 1: Baslik (Sol) ve Model Metrikleri (Sag) ---
        cv2.putText(final_canvas, "TUBITAK 2209-B | Akilli EV Sarj ve Park Yonetim Sistemi (V4)", 
                    (20, img_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Sag taraftaki model bilgisi (Inference ve FPS tamamen kaldirildi)
        right_info = "Model: YOLOv8m (1024x1024)"
        (rw, _), _ = cv2.getTextSize(right_info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(final_canvas, right_info, (img_w - 20 - rw, img_h + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 165, 180), 1, cv2.LINE_AA)

        # --- SATIR 2: Canli Durum Rozetleri (Sol) ve Varsa Uyari (Sag) ---
        pills = [
            (f"Bos: {counts['station_bosta']}", (110, 175, 30)),
            (f"Sarjda: {counts['station_sarj']}", (220, 150, 20)),
            (f"Ihlal: {counts['station_park']}", (20, 90, 220)),
            (f"Fis Yerde: {counts['Fis_yerde']}", (50, 50, 220) if counts["Fis_yerde"] > 0 else (120, 130, 140))
        ]

        # Rozetleri soldan baslayarak duzenli yerlestir
        cur_x = 20
        for text, dot_color in pills:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            pw = tw + 22
            # Rozet kutusu
            cv2.rectangle(final_canvas, (cur_x, img_h + 34), (cur_x + pw, img_h + 54), (32, 38, 48), -1)
            cv2.rectangle(final_canvas, (cur_x, img_h + 34), (cur_x + pw, img_h + 54), (55, 65, 80), 1)
            # Renkli nokta
            cv2.circle(final_canvas, (cur_x + 8, img_h + 44), 3, dot_color, -1, cv2.LINE_AA)
            # Metin
            cv2.putText(final_canvas, text, (cur_x + 16, img_h + 48), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
            cur_x += pw + 8

        # Eger aktif alarm varsa sag tarafa ferah sekilde yerlestir
        if alerts:
            alert_text = f"! UYARI: {alerts[0]}"
            (aw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            ax = img_w - 20 - aw - 16
            cv2.rectangle(final_canvas, (ax, img_h + 34), (img_w - 20, img_h + 54), (25, 20, 120), -1)
            cv2.rectangle(final_canvas, (ax, img_h + 34), (img_w - 20, img_h + 54), (50, 50, 220), 1)
            cv2.putText(final_canvas, alert_text, (ax + 8, img_h + 48), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 255), 1, cv2.LINE_AA)

        return final_canvas

    def process_video(self, source, save_path: str = None, show: bool = True):
        """Video veya webcam isle."""
        if isinstance(source, int) or source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print(f"[*] Kamera acildi: {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"[*] Video acildi: {source}")

        if not cap.isOpened():
            raise ValueError(f"Kaynak acilamadi: {source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height + 62))

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
                    disp_frame = cv2.resize(annotated, (1280, 720)) if width > 1280 else annotated
                    cv2.imshow("EV Sarj Istasyonu V4", disp_frame)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        break

        except KeyboardInterrupt:
            print("\n[!] Durduruldu (Ctrl+C)")
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
            print(f"\n[+] Islem tamamlandi: {frame_count} frame islendi")

    def process_image(self, image_path: str, save_path: str = None, show: bool = True):
        """Tek bir gorseli isle."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Gorsel okunamadi: {image_path}")
        
        result = self.process_frame(frame)
        perf = result["performance"]
        print(f"Inference: {perf['inference_ms']:.2f} ms | Toplam: {perf['total_ms']:.2f} ms")
        
        if save_path:
            cv2.imwrite(save_path, result["annotated_frame"])
            print(f"[+] Sonuc kaydedildi: {save_path}")
            
        if show:
            disp_frame = cv2.resize(result["annotated_frame"], (1280, 720)) if frame.shape[1] > 1280 else result["annotated_frame"]
            cv2.imshow("EV Sarj Istasyonu V4", disp_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="EV Sarj Istasyonu Akilli Yonetim Pipeline V4")
    parser.add_argument("--source", type=str, required=True, help="Kaynak: gorsel yolu, video yolu veya kamera index (0)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model yolu (.pt)")
    parser.add_argument("--conf", type=float, default=0.45, help="Minimum guven skoru")
    parser.add_argument("--save", type=str, default=None, help="Cikti kaydedilecek dosya yolu (.mp4 veya .jpg)")
    parser.add_argument("--no-show", action="store_true", help="Pencere gosterme")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not Path(args.model).exists():
        print(f"[-] Model bulunamadi: {args.model}")
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