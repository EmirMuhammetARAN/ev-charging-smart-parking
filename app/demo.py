"""
Gradio Demo Uygulamasi V4
=========================
EV Sarj Istasyonu Akilli Park Yonetimi ve Guvenlik Izleme Sistemi (V4)
TUBITAK 2209-B Prototip Demo Arayuzu.

Kullanim:
    python app/demo.py
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

# Proje kok dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "ev_charging_v4" / "weights" / "best.pt"

# Pipeline modulunu import et
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_v4 import EVChargingPipelineV4

CLASS_NAMES_TR = {
    "Fis_Bosta": "Fis Bosta",
    "Fis_Takili": "Fis Takili (Sarjda)",
    "Fis_yerde": "Kritik: Fis Yerde (Anomali!)",
    "car_charging": "Arac (Sarj Oluyor)",
    "car_parked": "Arac (Park Halinde / Isgal)",
    "station_bosta": "Istasyon (Musait / Bos)",
    "station_park": "Istasyon (Park Ihlali)",
    "station_sarj": "Istasyon (Aktif Sarj)"
}

pipeline_cache = {}

def get_pipeline(model_path: str = str(DEFAULT_MODEL), conf: float = 0.5):
    key = (model_path, conf)
    if key not in pipeline_cache:
        if not Path(model_path).exists():
            return None
        pipeline_cache[key] = EVChargingPipelineV4(model_path=model_path, confidence=conf)
    return pipeline_cache[key]

def detect_image(image: np.ndarray, confidence: float, model_path: str):
    if image is None:
        return None, "Lutfen bir gorsel yukleyin.", ""

    pipeline = get_pipeline(model_path, conf=confidence)
    if pipeline is None:
        return image, "Model yuklenemedi. Lutfen dosya yolunu kontrol edin.", ""

    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = pipeline.process_frame(frame_bgr)
    annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
    
    counts = result["counts"]
    alerts = result["alerts"]
    perf = result["performance"]
    detections = result["detections"]

    sarj_count = counts.get("station_sarj", 0)
    park_count = counts.get("station_park", 0)
    bos_count = counts.get("station_bosta", 0)
    fis_yerde = counts.get("Fis_yerde", 0)

    lines = ["### Sistem Analiz Ozeti\n"]
    lines.append(f"- **Inference Suresi:** {perf['inference_ms']:.1f} ms | **Anlik FPS:** {perf['fps']:.1f}")
    lines.append(f"- **Toplam Tespit Edilen Nesne:** {len(detections)}")
    lines.append(f"- **Aktif Sarj:** {sarj_count} istasyon")
    lines.append(f"- **Park Ihlali (Sarjsiz Bekleme):** {park_count} istasyon")
    lines.append(f"- **Musait Istasyon:** {bos_count} istasyon")
    if fis_yerde > 0:
        lines.append(f"- **Guvenlik Uyarisi:** {fis_yerde} adet kablo yerde!")
    lines.append("")

    if detections:
        lines.append("| Nesne Kodu | Turkce Tanim | Guven Skoru |")
        lines.append("|---|---|---|")
        for det in detections:
            cls_tr = CLASS_NAMES_TR.get(det["class_name"], det["class_name"])
            lines.append(f"| `{det['class_name']}` | {cls_tr} | %{det['confidence']*100:.1f} |")
    else:
        lines.append("*Herhangi bir nesne tespit edilemedi.*")

    result_text = "\n".join(lines)

    alert_badges = ""
    if alerts:
        for alert in alerts:
            alert_badges += f'<div style="background: #ef4444; color: white; padding: 10px 14px; border-radius: 6px; font-weight: bold; margin-bottom: 6px;">UYARI: {alert}</div>'
    else:
        alert_badges = '<div style="background: #10b981; color: white; padding: 10px 14px; border-radius: 6px; font-weight: bold;">Tum Durumlar Normal - Guvenlik Anomalisi Yok</div>'

    status_html = f'''
    <div style="background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 16px; color: white;">
        <h4 style="margin: 0 0 12px 0; color: #60a5fa;">TUBITAK 2209-B - Istasyon Durum Paneli</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin-bottom: 12px;">
            <div style="background: #111827; padding: 10px; border-radius: 8px; text-align: center; border-bottom: 3px solid #10b981;">
                <div style="font-size: 22px; font-weight: bold; color: #10b981;">{sarj_count}</div>
                <div style="font-size: 11px; color: #9ca3af;">SARJ OLUYOR</div>
            </div>
            <div style="background: #111827; padding: 10px; border-radius: 8px; text-align: center; border-bottom: 3px solid #f97316;">
                <div style="font-size: 22px; font-weight: bold; color: #f97316;">{park_count}</div>
                <div style="font-size: 11px; color: #9ca3af;">PARK IHLALI</div>
            </div>
            <div style="background: #111827; padding: 10px; border-radius: 8px; text-align: center; border-bottom: 3px solid #3b82f6;">
                <div style="font-size: 22px; font-weight: bold; color: #3b82f6;">{bos_count}</div>
                <div style="font-size: 11px; color: #9ca3af;">MUSAIT (BOS)</div>
            </div>
            <div style="background: #111827; padding: 10px; border-radius: 8px; text-align: center; border-bottom: 3px solid #ef4444;">
                <div style="font-size: 22px; font-weight: bold; color: #ef4444;">{fis_yerde}</div>
                <div style="font-size: 11px; color: #9ca3af;">FIS YERDE</div>
            </div>
        </div>
        <div>{alert_badges}</div>
    </div>
    '''

    return annotated_rgb, result_text, status_html

def detect_video(video_path: str, confidence: float, model_path: str):
    if not video_path:
        return None
        
    pipeline = get_pipeline(model_path, conf=confidence)
    if pipeline is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame)
        writer.write(result["annotated_frame"])

    cap.release()
    writer.release()
    return output_path

def create_demo(model_path: str = str(DEFAULT_MODEL), share: bool = False):
    with gr.Blocks(title="EV Sarj Istasyonu Akilli Yonetimi (V4)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # EV Sarj Istasyonlarinda Yapay Zeka Tabanli Akilli Park Yonetimi ve Guvenlik Izleme Sistemi (V4)
        ### TUBITAK 2209-B Projesi | Gazi Universitesi & Token Finansal Teknolojiler A.S.
        **Model:** YOLOv8m (1024x1024) | **Basarim:** %96.7 mAP50 | **Sinif:** 8 Sinifli Anomali ve Isgal Tespiti
        """)

        with gr.Tabs():
            with gr.Tab("Gorsel Analizi"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="Gorsel Yukle", type="numpy", height=400)
                        conf_slider = gr.Slider(minimum=0.1, maximum=0.95, value=0.5, step=0.05, label="Guven Esigi (Confidence)")
                        detect_btn = gr.Button("Analiz Et", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        img_output = gr.Image(label="YOLOv8m Tespit ve OSD Paneli", type="numpy", height=400)
                        status_html = gr.HTML(label="Sistem Durumu")

                results_md = gr.Markdown(label="Detayli Sonuclar")

                detect_btn.click(
                    fn=lambda img, conf: detect_image(img, conf, model_path),
                    inputs=[img_input, conf_slider],
                    outputs=[img_output, results_md, status_html],
                )

            with gr.Tab("Video / CCTV Analizi"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Video Yukle (.mp4)")
                        vid_conf = gr.Slider(minimum=0.1, maximum=0.95, value=0.5, step=0.05, label="Guven Esigi")
                        vid_btn = gr.Button("Videoyu Isle", variant="primary", size="lg")
                    with gr.Column():
                        vid_output = gr.Video(label="Gercek Zamanli OSD Islenmis Video")

                vid_btn.click(
                    fn=lambda vid, conf: detect_video(vid, conf, model_path),
                    inputs=[vid_input, vid_conf],
                    outputs=[vid_output],
                )

            with gr.Tab("Proje ve Model Bilgileri"):
                gr.Markdown(f"""
                ### Model Mimarisi ve Egitim Detaylari
                - **Mimari:** YOLOv8 Medium (YOLOv8m)
                - **Cozunurluk:** 1024x1024 (Ince kablo ve fis tespiti icin optimize edildi)
                - **Egitim Agirliklari:** `{model_path}`
                - **Epoch:** 100 Epoch (Cosine Annealing LR)
                - **mAP50:** %96.7 | **Precision:** %95.7 | **Recall:** %96.3
                
                ### Tespit Edilen 8 Sinif:
                1. `Fis_Bosta`: Sarj unitesi uzerinde bosta duran fis
                2. `Fis_Takili`: Araca takilmis sarj fisi
                3. `Fis_yerde`: Yerde birakilmis kablo/fis (**Kritik Is Guvenligi Anomalisi**)
                4. `car_charging`: Sarj alan aktif arac
                5. `car_parked`: Sarj etmeden park etmis arac (**Ihlal**)
                6. `station_bosta`: Kullanima hazir bos sarj yuvasi
                7. `station_park`: Sarjsiz isgal edilen istasyon alani
                8. `station_sarj`: Aktif sarj yapilan istasyon alani
                """)

    demo.launch(share=share, server_name="127.0.0.1", server_port=7860, show_error=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model yolu (.pt)")
    parser.add_argument("--share", action="store_true", help="Public link paylas")
    args = parser.parse_args()
    create_demo(model_path=args.model, share=args.share)

if __name__ == "__main__":
    main()
