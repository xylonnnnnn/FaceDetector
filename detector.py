#!/usr/bin/env python3
"""Standalone head detector for the improved YOLOv8 ONNX model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

try:
    import yaml
except ImportError:  # pragma: no cover - optional convenience dependency
    yaml = None


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "improved_head_model"
DEFAULT_MODEL = MODEL_DIR / "head_yolov8s_960_best.onnx"
DEFAULT_RKNN_MODEL = MODEL_DIR / "head_yolov8s_960_best.rknn"
DEFAULT_CONFIG = MODEL_DIR / "inference_config.yaml"
MODEL_PROFILES = {
    "quality": {
        "model": MODEL_DIR / "head_yolov8s_960_best.onnx",
        "imgsz": 960,
        "conf": 0.40,
        "description": "maximum accuracy",
    },
    "balanced": {
        "model": MODEL_DIR / "head_yolov8s_640_fast.onnx",
        "imgsz": 640,
        "conf": 0.45,
        "description": "better speed with useful accuracy",
    },
    "fast": {
        "model": MODEL_DIR / "head_yolov8s_384_fast.onnx",
        "imgsz": 384,
        "conf": 0.55,
        "description": "near real-time CPU mode",
    },
    "realtime": {
        "model": MODEL_DIR / "head_yolov8n_640_realtime.onnx",
        "imgsz": 640,
        "conf": 0.40,
        "description": "best CPU real-time balance",
    },
    "ultrafast": {
        "model": MODEL_DIR / "head_yolov8s_320_fast.onnx",
        "imgsz": 320,
        "conf": 0.60,
        "description": "strict 30 FPS CPU target with lower recall",
    },
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
DEFAULT_CAMERA_CONFIDENCE = None
DEFAULT_CAMERA_FPS = 30.0
PREFERRED_ONNX_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int = 0
    class_name: str = "head"
    track_id: int = -1

    def as_legacy_list(self) -> list[float]:
        """Old detector-compatible format: x1, y1, x2, y2, track_id, class_id, score."""
        return [self.x1, self.y1, self.x2, self.y2, self.track_id, self.class_id, self.score]


class SimpleIoUTracker:
    """Small dependency-free tracker for stable IDs in video/camera streams."""

    def __init__(self, iou_threshold: float = 0.35, max_missing: int = 12):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_id = 0
        self.tracks: dict[int, dict[str, np.ndarray | int]] = {}

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return float(inter / max(area_a + area_b - inter, 1e-6))

    def update(self, detections: list[Detection]) -> list[Detection]:
        for track in self.tracks.values():
            track["missing"] = int(track["missing"]) + 1

        unmatched_track_ids = set(self.tracks)
        for det in sorted(detections, key=lambda item: item.score, reverse=True):
            box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
            best_track_id = None
            best_iou = 0.0
            for track_id in unmatched_track_ids:
                iou = self._iou(box, self.tracks[track_id]["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                det.track_id = best_track_id
                self.tracks[best_track_id] = {"box": box, "missing": 0}
                unmatched_track_ids.remove(best_track_id)
            else:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {"box": box, "missing": 0}
                self.next_id += 1

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if int(track["missing"]) <= self.max_missing
        }
        return detections


class HeadDetector:
    """ONNXRuntime detector for the improved head-only model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        config_path: str | Path = DEFAULT_CONFIG,
        image_size: int | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        providers: list[str] | None = None,
        threads: int = 0,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        config = self._load_config(config_path)
        runtime = config.get("runtime", {})
        self.confidence = float(confidence if confidence is not None else runtime.get("confidence_threshold_balanced", 0.40))
        self.iou = float(iou if iou is not None else runtime.get("nms_iou_threshold", 0.70))
        self.names = {0: "head"}
        session_providers = providers or default_onnx_providers()
        session_options = create_session_options(threads)
        try:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=session_providers)
        except Exception as exc:
            if providers is not None or session_providers == ["CPUExecutionProvider"]:
                raise
            print(f"warning: failed to initialize ONNX providers {session_providers}; falling back to CPU: {exc}")
            self.session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = int(image_size or self._session_image_size() or runtime.get("image_size", 960))

    def _session_image_size(self) -> int | None:
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int) and shape[2] == shape[3]:
            return int(shape[2])
        return None

    @staticmethod
    def _load_config(config_path: str | Path) -> dict:
        path = Path(config_path)
        if not path.exists() or yaml is None:
            return {}
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: int | tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image, ratio, (dw, dh)

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        image, ratio, pad = self.letterbox(frame, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
        return image, ratio, pad

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if len(boxes) == 0:
            return []

        nms_boxes = boxes.copy()
        nms_boxes[:, 2] = nms_boxes[:, 2] - nms_boxes[:, 0]
        nms_boxes[:, 3] = nms_boxes[:, 3] - nms_boxes[:, 1]
        cv_indices = cv2.dnn.NMSBoxes(nms_boxes.tolist(), scores.tolist(), 0.0, iou_threshold)
        if len(cv_indices) > 0:
            return np.array(cv_indices).reshape(-1).astype(int).tolist()

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            ious = inter / np.maximum(union, 1e-6)
            order = order[1:][ious <= iou_threshold]

        return keep

    def postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        frame_shape: tuple[int, int, int],
    ) -> list[Detection]:
        pred = output[0] if output.ndim == 3 else output
        if pred.shape[0] <= 10 and pred.shape[1] > pred.shape[0]:
            pred = pred.T
        if pred.shape[1] < 5:
            raise ValueError(f"Unexpected ONNX output shape: {output.shape}")

        boxes = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        mask = scores >= self.confidence
        if not np.any(mask):
            return []

        boxes = self._xywh_to_xyxy(boxes[mask])
        scores = scores[mask]
        class_ids = class_ids[mask]
        keep = self._nms(boxes, scores, self.iou)

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        dw, dh = pad
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio

        height, width = frame_shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
        boxes = boxes.round().astype(np.int32)

        detections: list[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    class_id=int(class_id),
                    class_name=self.names.get(int(class_id), str(class_id)),
                )
            )
        return detections

    def predict(self, frame: np.ndarray) -> list[Detection]:
        image, ratio, pad = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: image})
        return self.postprocess(outputs[0], ratio, pad, frame.shape)

    def run(self, frame: np.ndarray) -> list[list[float]]:
        """Compatibility method with the old detector API."""
        return [det.as_legacy_list() for det in self.predict(frame)]

    @staticmethod
    def draw_results(frame: np.ndarray, detections: Iterable[Detection | list[float]]) -> np.ndarray:
        image = frame.copy()
        for item in detections:
            if isinstance(item, Detection):
                x1, y1, x2, y2 = item.x1, item.y1, item.x2, item.y2
                track_id, score = item.track_id, item.score
            else:
                x1, y1, x2, y2 = map(int, item[:4])
                track_id = int(item[4]) if len(item) > 4 else -1
                score = float(item[6]) if len(item) > 6 else 0.0

            color = (50, 215, 235)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"head {score:.2f}" if track_id < 0 else f"#{track_id} head {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(0, y1 - th - 8)
            cv2.rectangle(image, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
            cv2.putText(image, label, (x1 + 4, y_text + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        return image


class FaceDetector(HeadDetector):
    """Compatibility wrapper for code that used the old SIMPLE_FACE_DETECTOR class name."""

    def __init__(
        self,
        model_path_person: str | Path = DEFAULT_MODEL,
        net_size: int = 960,
        core_mask_person: int | None = None,
        **kwargs,
    ):
        _ = core_mask_person  # Kept only for old RKNN-style constructor compatibility.
        super().__init__(model_path=model_path_person, image_size=net_size, **kwargs)


class RknnHeadDetector(HeadDetector):
    """RKNNLite backend for RK3588 deployment."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RKNN_MODEL,
        config_path: str | Path = DEFAULT_CONFIG,
        image_size: int | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        core_mask: int | None = None,
    ):
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:  # pragma: no cover - RKNNLite is board-specific.
            raise ImportError(
                "RKNNLite is not installed. Install rknn-toolkit-lite2 on the RK3588 device "
                "or use the default ONNX backend."
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        config = self._load_config(config_path)
        runtime = config.get("runtime", {})
        self.image_size = int(image_size or runtime.get("image_size", 960))
        self.confidence = float(confidence if confidence is not None else runtime.get("confidence_threshold_balanced", 0.40))
        self.iou = float(iou if iou is not None else runtime.get("nms_iou_threshold", 0.70))
        self.names = {0: "head"}

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise OSError(f"{self.model_path}: load_rknn failed with code {ret}")
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_0
        ret = self.rknn.init_runtime(async_mode=True, core_mask=core_mask)
        if ret != 0:
            raise OSError(f"{self.model_path}: init_runtime failed with code {ret}")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        image, ratio, pad = self.letterbox(frame, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.uint8)
        return image, ratio, pad

    def predict(self, frame: np.ndarray) -> list[Detection]:
        image, ratio, pad = self.preprocess(frame)
        outputs = self.rknn.inference(inputs=[image], data_format="nhwc")
        return self.postprocess(outputs[0], ratio, pad, frame.shape)

    def close(self) -> None:
        self.rknn.release()


def source_to_capture_arg(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def create_session_options(threads: int = 0) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    if threads > 0:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return options


def default_onnx_providers() -> list[str]:
    available = ort.get_available_providers()
    providers = [provider for provider in PREFERRED_ONNX_PROVIDERS if provider in available]
    return providers or ["CPUExecutionProvider"]


def provider_list(name: str) -> list[str] | None:
    if name == "auto":
        return None
    mapping = {
        "cpu": ["CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    return mapping[name]


def is_camera_source(source: str) -> bool:
    return source.isdigit()


def list_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    return []


def save_crops(frame: np.ndarray, detections: list[Detection], crop_dir: Path, stem: str) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    for idx, det in enumerate(detections):
        crop = frame[det.y1 : det.y2, det.x1 : det.x2]
        if crop.size == 0:
            continue
        track = det.track_id if det.track_id >= 0 else idx
        cv2.imwrite(str(crop_dir / f"{stem}_track_{track}_score_{det.score:.2f}.jpg"), crop)


class LatestFrameCapture:
    """Continuously reads camera frames and keeps only the newest one."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.frame: np.ndarray | None = None
        self.frame_id = -1
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> "LatestFrameCapture":
        self.thread.start()
        return self

    def _reader(self) -> None:
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.stopped.set()
                break
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self) -> tuple[bool, np.ndarray | None, int]:
        while not self.stopped.is_set():
            with self.lock:
                if self.frame is not None:
                    return True, self.frame.copy(), self.frame_id
            time.sleep(0.001)
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy(), self.frame_id
        return False, None, self.frame_id

    def stop(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=1.0)


def process_images(args: argparse.Namespace, detector: HeadDetector) -> None:
    source = Path(args.source)
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"skip unreadable image: {image_path}")
            continue

        detections = detector.predict(frame)
        annotated = detector.draw_results(frame, detections)
        rel_name = image_path.name if source.is_file() else str(image_path.relative_to(source)).replace("/", "_")
        out_path = output_dir / rel_name
        cv2.imwrite(str(out_path), annotated)

        if args.save_crops:
            save_crops(frame, detections, Path(args.save_crops), image_path.stem)

        result = {
            "source": str(image_path),
            "output": str(out_path),
            "detections": [asdict(det) for det in detections],
        }
        all_results.append(result)
        print(f"{image_path}: {len(detections)} heads")

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")


def process_stream(args: argparse.Namespace, detector: HeadDetector) -> None:
    is_camera = is_camera_source(args.source)
    cap = cv2.VideoCapture(source_to_capture_arg(args.source))
    if not cap.isOpened():
        raise OSError(f"Cannot open source: {args.source}")

    if is_camera:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    latest_capture = LatestFrameCapture(cap).start() if is_camera and args.async_capture else None
    tracker = None if args.no_track else SimpleIoUTracker(args.track_iou, args.track_max_missing)
    writer = None
    json_results = []
    output_path = Path(args.output) if args.output else None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = args.camera_fps if is_camera else cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or np.isnan(fps):
            fps = DEFAULT_CAMERA_FPS if is_camera else 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    last_source_frame_id = -1
    t0 = time.perf_counter()
    try:
        while True:
            if latest_capture is not None:
                ok, frame, source_frame_id = latest_capture.read()
            else:
                ok, frame = cap.read()
                source_frame_id = frame_idx
            if not ok:
                break
            if frame is None:
                continue
            if latest_capture is not None and source_frame_id == last_source_frame_id:
                time.sleep(0.001)
                continue
            last_source_frame_id = source_frame_id

            detections = detector.predict(frame)
            if tracker is not None:
                detections = tracker.update(detections)

            annotated = detector.draw_results(frame, detections)
            if writer is not None:
                writer.write(annotated)

            if args.save_crops:
                save_crops(frame, detections, Path(args.save_crops), f"frame_{frame_idx:06d}")

            if args.save_json:
                json_results.append(
                    {
                        "frame": frame_idx,
                        "source_frame": source_frame_id,
                        "detections": [asdict(det) for det in detections],
                    }
                )

            if args.show:
                cv2.imshow("DETECTOR_v1 head detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            if frame_idx % 50 == 0:
                fps_now = frame_idx / max(time.perf_counter() - t0, 1e-6)
                print(f"processed={frame_idx} fps={fps_now:.2f} heads={len(detections)}")
    finally:
        if latest_capture is not None:
            latest_capture.stop()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone improved head detector")
    parser.add_argument("--source", default="0", help="Camera index, video path, image path, or image directory")
    parser.add_argument(
        "--profile",
        choices=["auto", "quality", "balanced", "fast", "realtime", "ultrafast"],
        default="auto",
        help="Speed/quality profile. auto uses realtime for camera and quality for files.",
    )
    parser.add_argument("--backend", choices=["auto", "onnx", "rknn"], default="auto", help="Inference backend")
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "coreml", "cuda", "tensorrt"],
        default="auto",
        help="ONNXRuntime execution provider",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to ONNX or RKNN model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to inference_config.yaml")
    parser.add_argument("--imgsz", type=int, default=None, help="Override model input size; default from config is 960")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold; default from config is 0.40")
    parser.add_argument(
        "--camera-conf",
        type=float,
        default=DEFAULT_CAMERA_CONFIDENCE,
        help="Camera-only confidence threshold used when --conf is not set; default is profile threshold",
    )
    parser.add_argument("--camera-fps", type=float, default=DEFAULT_CAMERA_FPS, help="Requested FPS for camera sources")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold; default from config is 0.70")
    parser.add_argument("--output", default=None, help="Output image directory or output video path")
    parser.add_argument("--show", action="store_true", help="Show live/window output")
    parser.add_argument("--save-json", default=None, help="Save detections to JSON")
    parser.add_argument("--save-crops", default=None, help="Directory for detected head crops")
    parser.add_argument("--async-capture", action=argparse.BooleanOptionalAction, default=True, help="Drop stale camera frames")
    parser.add_argument("--no-track", action="store_true", help="Disable lightweight tracking for video/camera")
    parser.add_argument("--track-iou", type=float, default=0.35, help="IoU threshold for lightweight tracking")
    parser.add_argument("--track-max-missing", type=int, default=12, help="Frames to keep unmatched tracks")
    parser.add_argument("--threads", type=int, default=0, help="CPU inference threads; 0 lets ONNXRuntime choose")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; useful for tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_path = Path(args.source)
    is_image_source = source_path.exists() and (source_path.is_dir() or source_path.suffix.lower() in IMAGE_EXTS)
    is_camera = is_camera_source(args.source) and not is_image_source
    profile_name = args.profile
    if profile_name == "auto":
        profile_name = "realtime" if is_camera else "quality"
    profile = MODEL_PROFILES[profile_name]

    backend = args.backend
    if backend == "auto":
        backend = "rknn" if str(args.model).lower().endswith(".rknn") else "onnx"

    default_model_selected = Path(args.model).resolve() == DEFAULT_MODEL.resolve()
    if backend == "onnx" and default_model_selected:
        args.model = str(profile["model"])
    if args.imgsz is None and backend == "onnx":
        args.imgsz = int(profile["imgsz"])
    if args.conf is None:
        confidence = args.camera_conf if is_camera and args.camera_conf is not None else float(profile["conf"])
    else:
        confidence = args.conf

    if backend == "rknn":
        if default_model_selected:
            args.model = str(DEFAULT_RKNN_MODEL)
        detector = RknnHeadDetector(
            model_path=args.model,
            config_path=args.config,
            image_size=args.imgsz,
            confidence=confidence,
            iou=args.iou,
        )
    else:
        detector = HeadDetector(
            model_path=args.model,
            config_path=args.config,
            image_size=args.imgsz,
            confidence=confidence,
            iou=args.iou,
            providers=provider_list(args.provider),
            threads=args.threads,
        )

    try:
        if is_image_source:
            if args.output is None:
                args.output = str(ROOT / "runs" / "images")
            process_images(args, detector)
        else:
            if args.output is None and not args.show:
                args.show = True
            process_stream(args, detector)
    finally:
        close = getattr(detector, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
