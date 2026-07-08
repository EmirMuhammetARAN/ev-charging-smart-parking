"""
Detection Pipeline
==================
3 aşamalı tespit pipeline'ı:
  1. YOLOv8 ile araç + kablo + soket tespiti
  2. OpenCV ile hareket analizi
  3. Durum takibi ve karar verme

Kullanım:
    # Tek görsel
    python scripts/pipeline.py --source image.jpg

    # Video
    python scripts/pipeline.py --source video.mp4

    # Webcam
    python scripts/pipeline.py --source 0

    # Çıktıyı kaydet
    python scripts/pipeline.py --source video.mp4 --save-video output.mp4
"""

import argparse
import time
import sys
from pathlib import Path
from enum import Enum
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL = RESULTS_DIR / "ev_charging" / "weights" / "best.pt"


# ============================================================
# Durum Tanımları
# ============================================================

class ParkingState(Enum):
    """Park yeri durumları."""
    EMPTY = "empty"                 # Park yeri boş
    OCCUPIED = "occupied"           # Araç var, kablo durumu bilinmiyor
    CHARGING = "charging"           # Araç var, kablo bağlı → şarj oluyor
    IDLE_OCCUPIED = "idle_occupied" # Araç var, kablo yok → şarj bitti ama çıkmıyor!
    LEAVING = "leaving"             # Araç ayrılıyor (hareket var)

    @property
    def display_name(self):
        names = {
            "empty": "Boş ✅",
            "occupied": "Araç Var",
            "charging": "Şarj Ediliyor 🔌",
            "idle_occupied": "⚠️ UYARI: Şarj Bitti!",
            "leaving": "Araç Ayrılıyor 🚗",
        }
        return names.get(self.value, self.value)

    @property
    def color(self):
        """BGR renkleri (OpenCV formatı)."""
        colors = {
            "empty": (0, 200, 0),          # Yeşil
            "occupied": (200, 200, 0),     # Cyan
            "charging": (200, 100, 0),     # Mavi
            "idle_occupied": (0, 0, 255),  # Kırmızı
            "leaving": (0, 200, 200),      # Sarı
        }
        return colors.get(self.value, (128, 128, 128))


# ============================================================
# Hareket Analizi
# ============================================================

class MotionAnalyzer:
    """
    Son N frame arasındaki farkı hesaplar.
    Background subtraction ile hareket tespiti yapar.
    """

    def __init__(self, history: int = 60, threshold: int = 25, min_area: int = 500):
        """
        Args:
            history: Background model için kullanılacak frame sayısı
            threshold: Piksel değişim eşiği
            min_area: Minimum hareket alanı (piksel²)
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=True,
        )
        self.min_area = min_area
        self.motion_history = deque(maxlen=30)  # Son 30 frame'in motion score'u

    def analyze(self, frame: np.ndarray, roi: tuple = None) -> float:
        """
        Frame'deki hareketi analiz et.

        Args:
            frame: BGR görsel (numpy array)
            roi: İlgi alanı (x1, y1, x2, y2). None ise tüm frame kullanılır.

        Returns:
            motion_score: 0.0 (hiç hareket yok) - 1.0 (çok hareket)
        """
        # ROI kes
        if roi is not None:
            x1, y1, x2, y2 = [int(v) for v in roi]
            analysis_frame = frame[y1:y2, x1:x2]
        else:
            analysis_frame = frame

        if analysis_frame.size == 0:
            return 0.0

        # Foreground mask
        fg_mask = self.bg_subtractor.apply(analysis_frame)

        # Gölgeleri temizle (gölge = 127, ön plan = 255)
        fg_mask[fg_mask == 127] = 0

        # Gürültü temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Motion score hesapla
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = analysis_frame.shape[0] * analysis_frame.shape[1]
        motion_score = motion_pixels / total_pixels if total_pixels > 0 else 0.0

        self.motion_history.append(motion_score)
        return motion_score

    @property
    def average_motion(self) -> float:
        """Son N frame'in ortalama motion score'u."""
        if not self.motion_history:
            return 0.0
        return sum(self.motion_history) / len(self.motion_history)

    @property
    def is_moving(self) -> bool:
        """Anlamlı hareket var mı?"""
        return self.average_motion > 0.02  # %2 threshold


# ============================================================
# Durum Takip Sistemi
# ============================================================

class StateTracker:
    """
    Her park yeri için durum makinesi.

    State transitions:
        EMPTY → OCCUPIED         (araç tespit edildi)
        OCCUPIED → CHARGING      (kablo bağlı tespit edildi)
        CHARGING → IDLE_OCCUPIED (kablo çıktı ama araç hâlâ orada)
        IDLE_OCCUPIED → LEAVING  (hareket tespit edildi)
        IDLE_OCCUPIED → ALERT    (X saniye geçti, hareket yok)
        ANY → EMPTY              (araç tespit edilmedi)
    """

    def __init__(self, alert_timeout: float = 60.0):
        """
        Args:
            alert_timeout: İdle durumunda kaç saniye sonra uyarı verileceği
        """
        self.state = ParkingState.EMPTY
        self.state_start_time = time.time()
        self.alert_timeout = alert_timeout
        self.idle_start_time = None

        # Uyarı sayacı
        self.alert_count = 0
        self.total_charging_time = 0.0
        self.total_idle_time = 0.0

    def update(
        self,
        vehicle_detected: bool,
        cable_detected: bool,
        is_moving: bool,
    ) -> ParkingState:
        """
        Yeni frame bilgisiyle durumu güncelle.

        Args:
            vehicle_detected: Araç tespit edildi mi?
            cable_detected: Kablo bağlı mı?
            is_moving: Hareket var mı?

        Returns:
            Güncel durum
        """
        now = time.time()
        prev_state = self.state

        if not vehicle_detected:
            # Araç yok → boş
            self.state = ParkingState.EMPTY
            self.idle_start_time = None

        elif cable_detected:
            # Araç var + kablo bağlı → şarj oluyor
            self.state = ParkingState.CHARGING
            self.idle_start_time = None

        elif is_moving:
            # Araç var + kablo yok + hareket → ayrılıyor
            self.state = ParkingState.LEAVING
            self.idle_start_time = None

        else:
            # Araç var + kablo yok + hareket yok → boşuna park ediyor!
            self.state = ParkingState.IDLE_OCCUPIED

            if self.idle_start_time is None:
                self.idle_start_time = now

        # Durum değişimi logla
        if self.state != prev_state:
            self.state_start_time = now

            if self.state == ParkingState.IDLE_OCCUPIED:
                self.alert_count += 1

        return self.state

    @property
    def idle_duration(self) -> float:
        """IDLE durumunda geçen süre (saniye)."""
        if self.idle_start_time is None:
            return 0.0
        return time.time() - self.idle_start_time

    @property
    def state_duration(self) -> float:
        """Mevcut durumda geçen süre (saniye)."""
        return time.time() - self.state_start_time

    @property
    def should_alert(self) -> bool:
        """Uyarı gönderilmeli mi?"""
        return (
            self.state == ParkingState.IDLE_OCCUPIED
            and self.idle_duration >= self.alert_timeout
        )


# ============================================================
# Ana Pipeline
# ============================================================

class EVChargingPipeline:
    """
    EV Şarj İstasyonu Akıllı Park Yönetimi Pipeline'ı.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        alert_timeout: float = 60.0,
    ):
        """
        Args:
            model_path: YOLOv8 model dosyası (.pt)
            confidence: Minimum güven skoru
            alert_timeout: İdle uyarı süresi (saniye)
        """
        print(f"🔄 Model yükleniyor: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Alt sistemler
        self.motion_analyzer = MotionAnalyzer()
        self.state_tracker = StateTracker(alert_timeout=alert_timeout)

        # Sınıf isimleri
        self.class_names = {0: "vehicle", 1: "charging_cable", 2: "charging_socket"}

        print("✅ Pipeline hazır!")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Tek bir frame'i işle.

        Args:
            frame: BGR görsel (numpy array)

        Returns:
            dict: {
                "detections": [...],
                "vehicle_detected": bool,
                "cable_detected": bool,
                "motion_score": float,
                "is_moving": bool,
                "state": ParkingState,
                "idle_duration": float,
                "should_alert": bool,
                "annotated_frame": np.ndarray,
            }
        """
        # 1. YOLOv8 ile nesne tespiti
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        vehicle_detected = False
        cable_detected = False
        vehicle_box = None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detection = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": xyxy.tolist(),  # [x1, y1, x2, y2]
                }
                detections.append(detection)

                if cls_id == 0:  # vehicle
                    vehicle_detected = True
                    vehicle_box = xyxy
                elif cls_id == 1:  # charging_cable
                    cable_detected = True

        # 2. Hareket analizi (araç bölgesinde)
        roi = vehicle_box if vehicle_detected else None
        motion_score = self.motion_analyzer.analyze(frame, roi=roi)
        is_moving = self.motion_analyzer.is_moving

        # 3. Durum güncelleme
        state = self.state_tracker.update(vehicle_detected, cable_detected, is_moving)

        # 4. Annotated frame oluştur
        annotated = self._annotate_frame(frame, detections, state, motion_score)

        return {
            "detections": detections,
            "vehicle_detected": vehicle_detected,
            "cable_detected": cable_detected,
            "motion_score": motion_score,
            "is_moving": is_moving,
            "state": state,
            "idle_duration": self.state_tracker.idle_duration,
            "should_alert": self.state_tracker.should_alert,
            "annotated_frame": annotated,
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list,
        state: ParkingState,
        motion_score: float,
    ) -> np.ndarray:
        """Frame üzerine tespit sonuçlarını çiz."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Bounding box'ları çiz
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Sınıfa göre renk
            colors = {
                "vehicle": (0, 255, 0),         # Yeşil
                "charging_cable": (255, 100, 0), # Mavi
                "charging_socket": (0, 200, 255), # Turuncu
            }
            color = colors.get(cls_name, (128, 128, 128))

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Durum paneli (sol üst)
        panel_h = 140
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (350, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        y_offset = 35
        cv2.putText(annotated, "EV Sarj Istasyonu Durumu",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset += 30
        state_color = state.color
        cv2.putText(annotated, f"Durum: {state.display_name}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        y_offset += 25
        cv2.putText(annotated, f"Arac: {'Var' if any(d['class_id']==0 for d in detections) else 'Yok'}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += 22
        cv2.putText(annotated, f"Kablo: {'Bagli' if any(d['class_id']==1 for d in detections) else 'Yok'}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += 22
        motion_text = f"Hareket: {motion_score:.3f}"
        cv2.putText(annotated, motion_text,
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # İdle uyarısı (büyük kırmızı yazı)
        if state == ParkingState.IDLE_OCCUPIED:
            idle_secs = self.state_tracker.idle_duration
            alert_text = f"SARJ BITTI! Bekleme: {idle_secs:.0f}s"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 40

            # Kırmızı arka plan
            cv2.rectangle(annotated, (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 200), -1)
            cv2.putText(annotated, alert_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return annotated

    def process_image(self, image_path: str) -> dict:
        """Tek bir görseli işle."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Görsel okunamadı: {image_path}")
        return self.process_frame(frame)

    def process_video(self, source, save_path: str = None, show: bool = True):
        """
        Video veya webcam işle.

        Args:
            source: Video dosyası yolu veya kamera index'i (0, 1, ...)
            save_path: Çıktı video yolu (None ise kaydetmez)
            show: Pencere göster
        """
        # Kaynak aç
        if isinstance(source, int) or source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print(f"📸 Kamera açıldı: {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"🎥 Video açıldı: {source}")

        if not cap.isOpened():
            raise ValueError(f"Kaynak açılamadı: {source}")

        # Video özellikleri
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"   Çözünürlük: {width}x{height}")
        print(f"   FPS: {fps}")
        if total_frames > 0:
            print(f"   Toplam frame: {total_frames}")

        # Video writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"   Çıktı: {save_path}")

        # Frame loop
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Pipeline
                result = self.process_frame(frame)
                annotated = result["annotated_frame"]

                # Kaydet
                if writer:
                    writer.write(annotated)

                # Göster
                if show:
                    cv2.imshow("EV Sarj Istasyonu - Pipeline", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # q veya ESC
                        break

                # Progress
                if frame_count % 100 == 0:
                    state = result["state"]
                    print(f"   Frame {frame_count}: {state.display_name}")

        except KeyboardInterrupt:
            print("\n⏹️  Durduruldu (Ctrl+C)")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        print(f"\n✅ İşlem tamamlandı: {frame_count} frame işlendi")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EV Şarj İstasyonu Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Kaynak: görsel yolu, video yolu veya kamera index (0)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Model yolu (.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum güven skoru")
    parser.add_argument("--alert-timeout", type=float, default=60.0,
                        help="İdle uyarı süresi (saniye)")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Çıktı video yolu")
    parser.add_argument("--no-show", action="store_true",
                        help="Pencere gösterme")
    return parser.parse_args()


def main():
    args = parse_args()

    print("🔌 EV Şarj İstasyonu — Detection Pipeline")
    print("=" * 50)

    # Model kontrolü
    if not Path(args.model).exists():
        print(f"❌ Model bulunamadı: {args.model}")
        print("   Önce train.py ile model eğitin.")
        sys.exit(1)

    # Pipeline oluştur
    pipeline = EVChargingPipeline(
        model_path=args.model,
        confidence=args.conf,
        alert_timeout=args.alert_timeout,
    )

    source = args.source

    # Kaynak türünü belirle
    if source.isdigit():
        # Webcam
        pipeline.process_video(source, save_path=args.save_video, show=not args.no_show)

    elif Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        # Tek görsel
        result = pipeline.process_image(source)

        print(f"\n📊 Sonuç:")
        print(f"   Durum: {result['state'].display_name}")
        print(f"   Araç: {'✅' if result['vehicle_detected'] else '❌'}")
        print(f"   Kablo: {'✅' if result['cable_detected'] else '❌'}")
        print(f"   Hareket: {result['motion_score']:.4f}")
        print(f"   Tespitler: {len(result['detections'])}")

        for det in result["detections"]:
            print(f"     {det['class_name']}: {det['confidence']:.3f} @ {det['bbox']}")

        # Annotated görseli kaydet
        output_path = Path(source).stem + "_result.jpg"
        cv2.imwrite(output_path, result["annotated_frame"])
        print(f"\n   Çıktı kaydedildi: {output_path}")

        # Göster
        if not args.no_show:
            cv2.imshow("Result", result["annotated_frame"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # Video dosyası
        pipeline.process_video(source, save_path=args.save_video, show=not args.no_show)


if __name__ == "__main__":
    main()
