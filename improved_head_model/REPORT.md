# Improved Head-Only Model

Дата обновления: 2026-05-07.

## Итог

Модель `improved_head_model` дообучена от предыдущего checkpoint на 18 дополнительных эпох: было 22 эпохи, стало эквивалентно 40 эпохам.

Принята не первая попытка: обычный fine-tune с warmup ухудшил `mAP50-95`, поэтому он был остановлен на 4-й эпохе. Финальная версия обучена аккуратным fine-tune режимом: `AdamW`, `lr0=0.0002`, `warmup_epochs=0`, `batch=12`, `imgsz=960`, `amp=False`.

Рекомендуемый рабочий режим остается: `imgsz=960`, `conf=0.40`, `iou=0.70`.

## Артефакты

| Файл | Назначение |
|---|---|
| `head_yolov8s_960_best.pt` | новый PyTorch checkpoint после fine-tune |
| `head_yolov8s_960_best.onnx` | новый ONNX export, input `1x3x960x960`, output `1x5x18900` |
| `head_yolov8s_960_best.rknn` | RKNN export текущей 40-epoch ONNX-модели для `rk3588` |
| `results.csv` | training metrics fine-tune запуска |
| `threshold_eval/threshold_sweep.csv` | FP/FN sweep новой модели на compact val split |

RKNN собран из текущего `head_yolov8s_960_best.onnx` через `rknn-toolkit2 2.3.2` для `rk3588` без INT8 quantization.

## Данные и обучение

| Параметр | Значение |
|---|---:|
| Base checkpoint | previous `head_yolov8s_960_best.pt` |
| Previous epochs | 22 |
| Extra fine-tune epochs | 18 |
| Total equivalent epochs | 40 |
| Train images | 12116 |
| Compact hard-negative images | 11529 |
| Extra high-resolution images | 587 |
| Train head object instances | 58685 |
| Validation images | 2299 |
| Validation head objects | 12164 |
| Image size | 960 |
| GPU | Tesla V100-SXM2-32GB |

High-resolution subset выбран из полного train split по сложным кадрам: много голов и/или маленькие головы. Это добавило 587 оригинальных изображений и 12090 head instances поверх compact hard-negative train.

## Validation Metrics

Сравнение выполнено на compact validation copy того же test split, чтобы обе модели оценивались на одинаковых изображениях и labels.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Previous epoch-22 model | 0.93622 | 0.93039 | 0.97035 | 0.73727 |
| New epoch-40 fine-tuned model | 0.94346 | 0.93966 | 0.97888 | 0.76083 |

Изменение:

| Метрика | Было | Стало | Изменение |
|---|---:|---:|---:|
| Precision | 0.93622 | 0.94346 | +0.00724 |
| Recall | 0.93039 | 0.93966 | +0.00927 |
| mAP50 | 0.97035 | 0.97888 | +0.00853 |
| mAP50-95 | 0.73727 | 0.76083 | +0.02356 |

## Threshold Sweep

Основной рабочий порог `conf=0.40`:

| Model | conf | TP | FP | FN | Precision | Recall | F1 | Background images with FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous epoch-22 model | 0.40 | 11346 | 868 | 818 | 0.9289 | 0.9328 | 0.9308 | 10 |
| New epoch-40 fine-tuned model | 0.40 | 11423 | 731 | 741 | 0.9399 | 0.9391 | 0.9395 | 6 |

Изменение при `conf=0.40`:

| Метрика | Было | Стало | Изменение |
|---|---:|---:|---:|
| TP | 11346 | 11423 | +77 |
| FP | 868 | 731 | -137 |
| FN | 818 | 741 | -77 |
| Precision | 0.9289 | 0.9399 | +0.0109 |
| Recall | 0.9328 | 0.9391 | +0.0063 |
| F1 | 0.9308 | 0.9395 | +0.0086 |

## Пороговые режимы новой модели

| conf | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 11503 | 838 | 661 | 0.9321 | 0.9457 | 0.9388 |
| 0.40 | 11423 | 731 | 741 | 0.9399 | 0.9391 | 0.9395 |
| 0.45 | 11339 | 637 | 825 | 0.9468 | 0.9322 | 0.9394 |
| 0.50 | 11228 | 556 | 936 | 0.9528 | 0.9231 | 0.9377 |

`conf=0.40` оставлен основным, потому что он дает лучший баланс F1 и одновременно улучшает FP/FN относительно epoch-22 модели.

## SHA256

| Artifact | SHA256 |
|---|---|
| `head_yolov8s_960_best.pt` | `197f7c088d412e6a91800fbe8273352c0c1b385cf8ad48ef7dc97fd97bba6504` |
| `head_yolov8s_960_best.onnx` | `f2fbd478c7250e01e8722e6c7f87e14f80d163b59788b9e44566e4b4538a7f7a` |
| `head_yolov8s_960_best.rknn` | `99245144807e4dc702ea67f2eb36d13d9afb29d1c06a99405b27ed5e15d9e248` |

## Важное

Текущий ONNX проверен на локальной загрузке через ONNXRuntime: input `1x3x960x960`, output `1x5x18900`.

Текущий RKNN собран и добавлен в пакет, но локальный запуск RKNN невозможен без `rknn-toolkit-lite2` на RK3588.
