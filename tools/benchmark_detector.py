#!/usr/bin/env python3
"""Benchmark DETECTOR_v1 profiles on a folder of images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import HeadDetector, MODEL_PROFILES, provider_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector profiles")
    parser.add_argument("--source", default="../data/images/test", help="Image folder or single image")
    parser.add_argument("--limit", type=int, default=80, help="Max images to benchmark")
    parser.add_argument("--provider", choices=["auto", "cpu", "coreml", "cuda", "tensorrt"], default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON report")
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in exts)


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        source = (ROOT / args.source).resolve()
    paths = image_paths(source)[: args.limit]
    frames = [cv2.imread(str(path)) for path in paths]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError(f"No readable images found in {source}")

    report = []
    for name, profile in MODEL_PROFILES.items():
        model = Path(profile["model"])
        if not model.exists():
            print(f"{name}: missing {model}")
            continue
        detector = HeadDetector(
            model_path=model,
            image_size=int(profile["imgsz"]),
            confidence=float(profile["conf"]),
            providers=provider_list(args.provider),
            threads=args.threads,
        )
        for frame in frames[: min(3, len(frames))]:
            detector.predict(frame)

        counts = []
        start = time.perf_counter()
        for frame in frames:
            counts.append(len(detector.predict(frame)))
        elapsed = time.perf_counter() - start
        row = {
            "profile": name,
            "model": str(model),
            "imgsz": int(profile["imgsz"]),
            "frames": len(frames),
            "fps": len(frames) / elapsed,
            "ms_per_frame": elapsed / len(frames) * 1000,
            "avg_detections": sum(counts) / len(counts),
            "providers": detector.session.get_providers(),
        }
        report.append(row)
        print(
            f"{name:8s} imgsz={row['imgsz']:3d} "
            f"fps={row['fps']:.2f} ms={row['ms_per_frame']:.1f} "
            f"avg_heads={row['avg_detections']:.2f}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
