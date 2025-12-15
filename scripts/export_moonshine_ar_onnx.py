#!/usr/bin/env python3
"""
Export UsefulSensors/moonshine-tiny-ar to ONNX encoder/decoder + tokenizer for React Native.

This avoids Optimum (which lacks Moonshine support) and instead uses torch.onnx.export
with the official remote code from the model repo (trust_remote_code=True).

Outputs (by default into ./assets/onnx-ar):
  - encoder.onnx                : encoder last_hidden_state
  - decoder.onnx                : decoder logits
  - tokenizer.json / preprocessor_config.json (copied from HF)

Quantization (optional --quantize int8) uses onnxruntime.quantization.quantize_dynamic.

Usage:
  python scripts/export_moonshine_ar_onnx.py --output-dir ./assets/onnx-ar --seconds 6 --quantize int8

Requirements:
  pip install torch transformers onnx onnxruntime
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def export_encoder(model, sample_inputs, out_path: Path):
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_values, attention_mask):
            return self.inner.model.encoder(
                input_values=input_values, attention_mask=attention_mask
            ).last_hidden_state

    encoder = EncoderWrapper(model)
    encoder.eval()
    input_values = sample_inputs["input_values"]
    attention_mask = sample_inputs["attention_mask"]

    torch.onnx.export(
        encoder,
        (input_values, attention_mask),
        out_path,
        input_names=["input_values", "attention_mask"],
        output_names=["encoder_hidden_states"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={
            "input_values": {1: "audio_len"},
            "attention_mask": {1: "audio_len"},
            "encoder_hidden_states": {1: "time"},
        },
    )
    print(f"✔ Encoder ONNX saved: {out_path}")


def export_decoder(model, encoder_hidden, out_path: Path):
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, decoder_input_ids, encoder_hidden_states):
            dec = self.inner.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                use_cache=False,
            ).last_hidden_state
            return self.inner.proj_out(dec)

    decoder = DecoderWrapper(model)
    decoder.eval()

    vocab = model.config.vocab_size
    sample_tokens = torch.zeros((1, 4), dtype=torch.long)
    sample_enc = torch.zeros_like(encoder_hidden)
    torch.onnx.export(
        decoder,
        (sample_tokens, sample_enc),
        out_path,
        input_names=["decoder_input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={
            "decoder_input_ids": {1: "seq"},
            "encoder_hidden_states": {1: "time"},
            "logits": {1: "seq", 2: "vocab"},
        },
    )
    print(f"✔ Decoder ONNX saved: {out_path} (vocab={vocab})")


def quantize_int8(path: Path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        q_path = path.with_name(path.stem + "_int8.onnx")
        quantize_dynamic(
            str(path),
            str(q_path),
            weight_type=QuantType.QInt8,
            optimize_model=False,
        )
        print(f"✔ Quantized -> {q_path.name}")
    except Exception as e:
        print(f"Quantization skipped for {path.name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="UsefulSensors/moonshine-tiny-ar")
    parser.add_argument("--output-dir", default="./assets/onnx-ar")
    parser.add_argument("--seconds", type=float, default=6.0, help="Dummy audio length for tracing")
    parser.add_argument("--quantize", choices=["int8", None], default="int8")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    sr = processor.feature_extractor.sampling_rate
    sample_len = int(args.seconds * sr)
    dummy_audio = torch.zeros((1, sample_len), dtype=torch.float32)
    sample_inputs = processor(audio=dummy_audio.numpy(), sampling_rate=sr, return_tensors="pt")

    print("Running encoder once to get shape...")
    with torch.no_grad():
        enc_out = model.model.encoder(
            input_values=sample_inputs["input_values"],
            attention_mask=sample_inputs["attention_mask"],
        ).last_hidden_state
    print("Encoder hidden shape:", tuple(enc_out.shape))

    encoder_path = out_dir / "encoder.onnx"
    decoder_path = out_dir / "decoder.onnx"

    export_encoder(model, sample_inputs, encoder_path)
    export_decoder(model, enc_out, decoder_path)

    if args.quantize == "int8":
        quantize_int8(encoder_path)
        quantize_int8(decoder_path)

    # copy tokenizer files
    processor.save_pretrained(out_dir)
    print("✔ Tokenizer/preprocessor saved.")

    print("\nExport complete. Point ONNX_CONFIG in App.tsx to files in", out_dir)


if __name__ == "__main__":
    main()
