"""Convert FaceNet PyTorch model (.pth) to TensorFlow Lite (.tflite).

Pipeline:
1) Load PyTorch model and export to ONNX
2) Convert ONNX -> TensorFlow SavedModel
3) Convert SavedModel -> TFLite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import torch
from embedding import load_model
from onnx_tf.backend import prepare
import onnx


def convert(
    model_path: Path,
    onnx_path: Path,
    saved_model_dir: Path,
    tflite_path: Path,
    opset: int = 13,
) -> Path:
    model = load_model(model_path, device="cpu")

    dummy = torch.randn(1, 3, 160, 160)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    return tflite_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Convert .pth FaceNet model to .tflite")
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--onnx", type=Path, default=Path("build/model.onnx"))
    parser.add_argument("--saved-model", type=Path, default=Path("build/saved_model"))
    parser.add_argument("--output", type=Path, default=Path("model.tflite"))
    args = parser.parse_args()

    output = convert(args.model, args.onnx, args.saved_model, args.output)
    print(f"TFLite model saved at: {output}")


if __name__ == "__main__":
    cli()
