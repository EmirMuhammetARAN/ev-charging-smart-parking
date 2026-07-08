"""
Gradio Demo Uygulaması V3
=======================
EV Şarj İstasyonu Akıllı Park Yönetimi demo arayüzü.
Saf Yapay Zeka (End-to-End) V3 Sürümü.

Kullanım:
    python app/demo.py
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MODEL = RESULTS_DIR / "ev_charging_v3" / "weights" / "best.pt"

# Pipeline modülünü import et
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
try:
    from smart_analysis_v3 import draw_analysis
except ImportError:
    print("UYARI: smart_analysis_v3.py bulunamadı. Lütfen scripts klasöründe olduğundan emin olun.")
    draw_analysis = None

CLASS_NAMES_TR = {
    0: "Fiş Boşta", 
    1: "Fiş Takılı", 
    2: "Araç (Şarjda)", 
    3: "Araç (Park Halinde)", 
    4: "İstasyon (Boş)", 
    5: "İstasyon (Park)", 
    6: "İstasyon (Şarj)"
}


def load_model(model_path: str):
    """Modeli yükle."""
    path = Path(model_path)
    if not path.exists():
        return None
    return YOLO(str(path))


def detect_image(image: np.ndarray, confidence: float, model) -> tuple:
    """
    Tek bir görselde tespit yap.
    Returns: (annotated_image, results_text, status_html)
    """
    if image is None:
        return None, "Lütfen bir görsel yükleyin.", ""

    if model is None:
        return image, "❌ Model yüklenemedi. Önce eğitim yapın.", ""

    # YOLOv8 ile tespit
    results = model(image, conf=confidence, verbose=False)

    detections = []
    if results:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_name = results[0].names[cls_id]
            
            detections.append({
                "class": cls_name,
                "conf": conf,
                "box": xyxy
            })

    # Analiz çizimi (smart_analysis_v3 içindeki çizim modülü)
    if draw_analysis is not None:
        annotated = draw_analysis(image, detections)
    else:
        annotated = results[0].plot() if results else image

    cars = [d for d in detections if "car" in d["class"]]
    stations = [d for d in detections if "station" in d["class"]]
    
    sarj_count = sum(1 for s in stations if s["class"] == "station_sarj")
    park_count = sum(1 for s in stations if s["class"] == "station_park")
    bos_count = sum(1 for s in stations if s["class"] == "station_bosta")

    # Sonuç metni
    result_lines = ["### 📊 Analiz Özeti\n"]
    result_lines.append(f"- **Toplam İstasyon:** {len(stations)}")
    result_lines.append(f"- **Toplam Araç:** {len(cars)}")
    result_lines.append(f"- **Şarj Eden İstasyon:** {sarj_count}")
    result_lines.append(f"- **Sadece Park (Şarjsız):** {park_count}")
    result_lines.append(f"- **Boş İstasyon:** {bos_count}\n")
    
    if detections:
        result_lines.append("| Tespit Edilen | Güven |")
        result_lines.append("|-------|-------|")
        for det in detections:
            cls_idx = list(results[0].names.values()).index(det["class"])
            cls_name_tr = CLASS_NAMES_TR.get(cls_idx, det["class"])
            result_lines.append(f"| {cls_name_tr} | {det['conf']:.1%} |")
    else:
        result_lines.append("*Hiçbir nesne tespit edilmedi.*")

    result_text = "\n".join(result_lines)

    # Durum HTML Paneli
    status_html = f"""
    <div style="
        background: linear-gradient(135deg, #111827, #1f2937);
        border-left: 4px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
        color: white;
    ">
        <h3 style="margin: 0 0 12px 0; color: #60a5fa;">TÜBİTAK 2209-B - Otopark Durumu</h3>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="background: #374151; padding: 10px 15px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #10b981;">{sarj_count}</div>
                <div style="font-size: 12px; color: #9ca3af;">ŞARJ OLUYOR</div>
            </div>
            <div style="background: #374151; padding: 10px 15px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #f87171;">{park_count}</div>
                <div style="font-size: 12px; color: #9ca3af;">PARK HALİNDE</div>
            </div>
            <div style="background: #374151; padding: 10px 15px; border-radius: 6px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #d1d5db;">{bos_count}</div>
                <div style="font-size: 12px; color: #9ca3af;">MÜSAİT (BOŞ)</div>
            </div>
        </div>
    </div>
    """

    return annotated, result_text, status_html


def detect_video(video_path: str, confidence: float, model) -> str:
    """
    Video üzerinde tespit yap.
    """
    if video_path is None or model is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence, verbose=False)
        detections = []
        if results:
            for box in results[0].boxes:
                detections.append({
                    "class": results[0].names[int(box.cls[0].cpu().numpy())],
                    "conf": float(box.conf[0].cpu().numpy()),
                    "box": box.xyxy[0].cpu().numpy().tolist()
                })
                
        if draw_analysis is not None:
            annotated = draw_analysis(frame, detections)
        else:
            annotated = results[0].plot() if results else frame

        writer.write(annotated)

    cap.release()
    writer.release()
    return output_path


def get_model_info(model) -> str:
    """Model hakkında bilgi döndür."""
    if model is None:
        return "### ❌ Model Yüklenmedi\nModel dosyası bulunamadı."
    
    return """
### 🚀 Yapay Zeka Modeli Aktif
Bu sistem tamamen **Saf Yapay Zeka (End-to-End)** yaklaşımıyla çalışmaktadır.
Hiçbir eşleştirme algoritması (Distance, IoU) kullanılmadan model doğrudan durum tespiti yapar.
    """


def create_demo(model_path: str, share: bool = False):
    """Gradio arayüzünü oluştur ve başlat."""
    model = load_model(model_path)

    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    """

    with gr.Blocks(title="EV Şarj Akıllı Yönetim V3", theme=gr.themes.Soft(), css=custom_css) as demo:

        gr.Markdown("""
        # 🔌 EV Şarj İstasyonu — Saf AI (V3)
        ### Algoritmasız %100 Uçtan Uca Yapay Zeka Tespiti
        **TÜBİTAK 2209-B** | Gazi Üniversitesi
        """)

        with gr.Tabs():
            with gr.Tab("📸 Görsel Tespiti"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="Görsel Yükle", type="numpy", height=400)
                        conf_slider = gr.Slider(minimum=0.1, maximum=0.95, value=0.5, step=0.05, label="Güven Eşiği (Confidence)")
                        detect_btn = gr.Button("🔍 Tespit Et", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        img_output = gr.Image(label="Tespit Sonucu", type="numpy", height=400)
                        status_html = gr.HTML(label="Genel Durum")

                results_md = gr.Markdown(label="Detaylı Sonuçlar")

                detect_btn.click(
                    fn=lambda img, conf: detect_image(img, conf, model),
                    inputs=[img_input, conf_slider],
                    outputs=[img_output, results_md, status_html],
                )

            with gr.Tab("🎥 Video Tespiti"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Video Yükle")
                        vid_conf = gr.Slider(minimum=0.1, maximum=0.95, value=0.5, step=0.05, label="Güven Eşiği")
                        vid_btn = gr.Button("🎬 Videoyu İşle", variant="primary", size="lg")
                    with gr.Column():
                        vid_output = gr.Video(label="İşlenmiş Video")

                vid_btn.click(
                    fn=lambda vid, conf: detect_video(vid, conf, model),
                    inputs=[vid_input, vid_conf],
                    outputs=[vid_output],
                )

            with gr.Tab("📊 Model Bilgileri"):
                model_info = gr.Markdown(value=get_model_info(model))

    demo.launch(share=share, server_name="127.0.0.1", server_port=7860, show_error=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model yolu (.pt)")
    parser.add_argument("--share", action="store_true", help="Public URL ile paylaş")
    return parser.parse_args()


def main():
    args = parse_args()
    print("🔌 EV Şarj İstasyonu — Gradio Demo V3")
    if not Path(args.model).exists():
        print(f"⚠️ Model bulunamadı: {args.model}")
    create_demo(model_path=args.model, share=args.share)


if __name__ == "__main__":
    main()
