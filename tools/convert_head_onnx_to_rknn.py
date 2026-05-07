#!/usr/bin/env python3
"""Convert the improved head ONNX model to RKNN for RK3588."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX -> RKNN converter for head_yolov8s_960_best")
    parser.add_argument("--onnx", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output RKNN model")
    parser.add_argument("--target", default="rk3588", help="RKNN target platform")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--dataset", default=None, help="Calibration dataset file for INT8 quantization")
    parser.add_argument("--verbose", action="store_true", help="Verbose RKNN logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    if args.quantize and not args.dataset:
        raise ValueError("--dataset is required when --quantize is enabled")

    rknn = RKNN(verbose=args.verbose)
    try:
        print(f"Config target={args.target}")
        ret = rknn.config(
            target_platform=args.target,
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        print(f"Load ONNX: {onnx_path}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        print(f"Build RKNN quantize={args.quantize}")
        ret = rknn.build(do_quantization=args.quantize, dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Export RKNN: {output_path}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")

        print("done")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
