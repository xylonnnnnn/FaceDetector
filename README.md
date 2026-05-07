# DETECTOR_v1

Standalone head detector with ONNXRuntime backend and the improved head-only YOLOv8 model.

The default quality model is trained for `imgsz=960` and can be run on CPU, CUDA, TensorRT or CoreML through ONNXRuntime. The folder is self-contained for ONNX inference: clone/download it, install requirements, run the detector.

## Contents

| Path | Purpose |
|---|---|
| `detector.py` | CLI + Python API detector |
| `improved_head_model/head_yolov8s_960_best.onnx` | Main quality ONNX model, `1x3x960x960 -> 1x5x18900` |
| `improved_head_model/head_yolov8s_960_best.pt` | PyTorch checkpoint for retraining/export |
| `improved_head_model/head_yolov8s_960_best.rknn` | RK3588 RKNN export of the current 40-epoch model |
| `improved_head_model/head_yolov8s_640_fast.onnx` | Faster 640 profile |
| `improved_head_model/head_yolov8s_384_fast.onnx` | Faster 384 profile |
| `improved_head_model/head_yolov8s_320_fast.onnx` | Ultrafast 320 profile |
| `improved_head_model/head_yolov8n_640_realtime.onnx` | Realtime camera profile |
| `improved_head_model/inference_config.yaml` | Runtime thresholds, hashes, training metadata |
| `improved_head_model/REPORT.md` | Model metrics and training report |
| `examples/faces_4_14_0029.png` | Small image for smoke testing |
| `tools/benchmark_detector.py` | Local FPS benchmark |
| `tools/convert_head_onnx_to_rknn.py` | Optional ONNX -> RKNN converter |
| `requirements.txt` | Minimal Python dependencies |
| `run.sh` | Convenience launcher |

## Install

Use Python 3.10-3.12. Some `onnxruntime` wheels may not be available for newer Python versions.

```bash
cd DETECTOR_v1
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

For NVIDIA GPU inference, install an ONNXRuntime GPU build suitable for your CUDA environment instead of the CPU-only `onnxruntime` package.

## Smoke Test

Run the bundled example image:

```bash
./run.sh --source examples/faces_4_14_0029.png --output outputs/example --save-json outputs/example.json --profile quality
```

Expected behavior: the command creates an annotated image in `outputs/example/` and a JSON file with detected heads.

Check that the model loads:

```bash
python3 - <<'PY'
from detector import HeadDetector
d = HeadDetector()
print(d.model_path)
print(d.image_size)
print(d.session.get_providers())
PY
```

## Usage

Image or folder:

```bash
./run.sh --source /path/to/image.jpg --output outputs/images --save-json outputs/images.json
./run.sh --source /path/to/images --output outputs/folder --save-json outputs/folder.json
```

Video:

```bash
./run.sh --source /path/to/video.mp4 --output outputs/video_result.mp4 --save-json outputs/video.json
```

Camera:

```bash
./run.sh --source 0 --show
```

For camera sources, `--profile auto` selects `realtime`, uses asynchronous capture, and requests `30 FPS`.

## Profiles

| Profile | Model | Input | Purpose |
|---|---|---:|---|
| `quality` | `head_yolov8s_960_best.onnx` | 960 | Best accuracy |
| `balanced` | `head_yolov8s_640_fast.onnx` | 640 | Speed/quality compromise |
| `fast` | `head_yolov8s_384_fast.onnx` | 384 | Faster CPU mode |
| `realtime` | `head_yolov8n_640_realtime.onnx` | 640 | Camera default |
| `ultrafast` | `head_yolov8s_320_fast.onnx` | 320 | Highest CPU FPS, lower recall |

Manual profile selection:

```bash
./run.sh --profile quality --source examples/faces_4_14_0029.png --output outputs/quality
./run.sh --profile realtime --source 0 --show
```

## Thresholds

Recommended quality mode:

```bash
./run.sh --source /path/to/image.jpg --output outputs/images --conf 0.40 --iou 0.70 --imgsz 960
```

Use a stricter camera threshold if false positives are more expensive:

```bash
./run.sh --source 0 --show --camera-conf 0.55 --camera-fps 30
```

For the current `head_yolov8s_960_best.onnx` model at `conf=0.40` on the validation split: `TP=11423`, `FP=731`, `FN=741`, `precision=0.9399`, `recall=0.9391`, `F1=0.9395`.

## Python API

```python
import cv2
from detector import HeadDetector

detector = HeadDetector()
frame = cv2.imread("examples/faces_4_14_0029.png")
detections = detector.predict(frame)

for det in detections:
    print(det.x1, det.y1, det.x2, det.y2, det.score)
```

Compatibility API for code that used `SIMPLE_FACE_DETECTOR.FaceDetector`:

```python
from detector import FaceDetector

detector = FaceDetector(
    model_path_person="improved_head_model/head_yolov8s_960_best.onnx",
    net_size=960,
)
result = detector.run(frame)
```

`run(frame)` returns the legacy list format:

```text
[x1, y1, x2, y2, track_id, class_id, score]
```

## Benchmark

```bash
python3 tools/benchmark_detector.py --source examples --limit 1 --provider cpu
```

For a more representative benchmark, pass a folder with many images.

## RKNN

RKNN is bundled for the current 40-epoch model:

```text
improved_head_model/head_yolov8s_960_best.rknn
```

It was built from `head_yolov8s_960_best.onnx` for `rk3588` without INT8 quantization. SHA256:

```text
99245144807e4dc702ea67f2eb36d13d9afb29d1c06a99405b27ed5e15d9e248
```

Run on RK3588 with `rknn-toolkit-lite2` installed:

```bash
./run.sh --backend rknn --model improved_head_model/head_yolov8s_960_best.rknn --source 0 --show
```

To rebuild RKNN on a machine with `rknn-toolkit2`:

```bash
python3 tools/convert_head_onnx_to_rknn.py \
  --onnx improved_head_model/head_yolov8s_960_best.onnx \
  --output improved_head_model/head_yolov8s_960_best.rknn \
  --target rk3588
```

## Difference From SIMPLE_FACE_DETECTOR

`SIMPLE_FACE_DETECTOR` was RKNN-only and RK3588-oriented. It required `rknnlite`, SFSORT, `/dev/video0` camera assumptions, and board-specific utilities.

`DETECTOR_v1` is portable by default: ONNXRuntime works on a normal laptop/server, supports images, folders, video and camera input, includes a lightweight built-in tracker, and preserves the old `FaceDetector.run(frame)` return format for integration compatibility.
