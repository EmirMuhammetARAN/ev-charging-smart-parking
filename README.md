# EV Charging Station Smart Parking Management

TUBITAK 2209-B project: AI-based smart parking management and monitoring for electric vehicle charging stations.

## Project Summary

This repository contains a computer vision prototype for detecting the status of EV charging station parking areas. The final system uses a YOLOv8m object detection model trained on a synthetic, simulation-based dataset prepared for EV charging station scenarios.

The final model detects 8 classes:

| ID | Class |
| --- | --- |
| 0 | Fis_Bosta |
| 1 | Fis_Takili |
| 2 | Fis_yerde |
| 3 | car_charging |
| 4 | car_parked |
| 5 | station_bosta |
| 6 | station_park |
| 7 | station_sarj |

## Final Dataset

The final dataset is `datasets/dataset v3`.

| Split | Image count |
| --- | ---: |
| Train | 195 |
| Validation | 48 |
| Total | 243 |

The dataset includes synthetic EV charging station scenes generated with NVIDIA Isaac Sim and labeled in YOLO format through Roboflow. Night and fog scenarios are included in the dataset, while rain robustness is evaluated with image augmentation.

## Final Training Run

The final model is trained with:

| Item | Value |
| --- | --- |
| Model | YOLOv8m |
| Run directory | `runs/ev_charging_v4` |
| Weights | `runs/ev_charging_v4/weights/best.pt` |
| Epochs | 100 |
| Image size | 1024 |
| Batch size | 8 |
| Dataset | `datasets/dataset v3/data.yaml` |

Final validation metrics from `runs/ev_charging_v4/results.csv`:

| Metric | Result |
| --- | ---: |
| Precision | 95.7% |
| Recall | 96.3% |
| mAP50 | 96.7% |
| mAP50-95 | 82.7% |

## Main Files

| Path | Purpose |
| --- | --- |
| `scripts/isaac_sim_auto_capture.py` | Synthetic data generation in NVIDIA Isaac Sim |
| `scripts/augment_weather.py` | Night, fog, and rain augmentation utilities |
| `scripts/train_v4.py` | Final YOLOv8m training script |
| `scripts/pipeline_v4.py` | Smart analysis pipeline with alerts, dashboard, FPS, and inference timing |
| `runs/ev_charging_v4/` | Final training outputs and evaluation plots |

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the final model:

```bash
python scripts/train_v4.py
```

Run the smart pipeline on an image:

```bash
python scripts/pipeline_v4.py --source path/to/image.jpg --save output.jpg --no-show
```

Run the smart pipeline on a video or webcam:

```bash
python scripts/pipeline_v4.py --source path/to/video.mp4
python scripts/pipeline_v4.py --source 0
```

The V4 pipeline reports per-frame inference time and FPS on the dashboard and prints average inference/FPS values after video processing.

## Evaluation Artifacts

Important output files are available under `runs/ev_charging_v4`:

- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `BoxPR_curve.png`
- `BoxF1_curve.png`
- `val_batch*_pred.jpg`

## Notes

The current model is trained and validated on synthetic data. Real charging station camera footage should be used in future work for domain adaptation and field validation.
