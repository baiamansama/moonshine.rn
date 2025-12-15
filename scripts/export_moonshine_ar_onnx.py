#!/usr/bin/env python3
"""
Export Moonshine-tiny-ar to ONNX format for React Native.

This script exports UsefulSensors/moonshine-tiny-ar model to ONNX format
with optional int8 quantization for mobile deployment.

Requirements:
    pip install transformers optimum[exporters] onnx onnxruntime

Usage:
    python scripts/export_moonshine_ar_onnx.py --output-dir ./assets/onnx-ar
    python scripts/export_moonshine_ar_onnx.py --output-dir ./assets/onnx-ar --quantize int8

References:
    - Model: https://huggingface.co/UsefulSensors/moonshine-tiny-ar
    - ONNX Export: https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
"""

import argparse
import json
import os
import shutil
from pathlib import Path

def export_moonshine_onnx(
    model_id: str = "UsefulSensors/moonshine-tiny-ar",
    output_dir: str = "./assets/onnx-ar",
    quantize: str = None,
):
    """Export Moonshine model to ONNX format."""

    print(f"Exporting {model_id} to ONNX...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        from optimum.exporters.onnx import main_export
        from optimum.exporters.onnx.model_configs import MoonshineOnnxConfig
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "optimum[exporters]", "onnx", "onnxruntime"], check=True)
        from optimum.exporters.onnx import main_export

    # Export to ONNX using optimum
    print("Running ONNX export...")
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="automatic-speech-recognition",
        opset=14,  # Good compatibility
        device="cpu",
        fp16=False,  # Use fp32 for compatibility
        optimize="O1",  # Basic optimization
    )

    # Quantize if requested
    if quantize == "int8":
        print("Applying int8 quantization...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            onnx_files = list(Path(output_dir).glob("*.onnx"))
            for onnx_file in onnx_files:
                if "_int8" not in onnx_file.name:
                    output_path = onnx_file.with_suffix("").with_suffix("_int8.onnx")
                    print(f"Quantizing {onnx_file.name} -> {output_path.name}")
                    quantize_dynamic(
                        str(onnx_file),
                        str(output_path),
                        weight_type=QuantType.QInt8,
                    )
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing with fp32 models...")

    # Copy tokenizer files
    print("Copying tokenizer files...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")

    # List exported files
    print("\nExported files:")
    for f in sorted(Path(output_dir).iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

    print("\nDone! Copy the ONNX files to your React Native app's assets folder.")
    print("Update App.tsx to point to the new Arabic model files.")


def main():
    parser = argparse.ArgumentParser(
        description="Export Moonshine-tiny-ar to ONNX for React Native"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="UsefulSensors/moonshine-tiny-ar",
        help="HuggingFace model ID (default: UsefulSensors/moonshine-tiny-ar)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./assets/onnx-ar",
        help="Output directory for ONNX files (default: ./assets/onnx-ar)"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int8", None],
        default=None,
        help="Quantization type (optional, default: none)"
    )

    args = parser.parse_args()
    export_moonshine_onnx(
        model_id=args.model_id,
        output_dir=args.output_dir,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
