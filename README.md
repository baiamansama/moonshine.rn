# Moonshine ASR - React Native

On-device Arabic/English speech-to-text using Moonshine models with ONNX Runtime.

## Status: ONNX Runtime Pipeline Ready

- Using `onnxruntime-react-native` for cross-platform inference
- English model (int8 quantized) included for pipeline validation
- Arabic export script ready - just run it to get Arabic models

## Why Moonshine Arabic?

| Model | Params | Arabic WER |
|-------|--------|------------|
| **moonshine-tiny-ar** | 27M | **20.76** |
| whisper-tiny | 39M | 66.01 |

3x more accurate than Whisper Tiny for Arabic.

## Stack

- Expo SDK 54 / RN 0.81 / New Architecture
- onnxruntime-react-native 1.23.2
- react-native-audio-api (16kHz capture)
- react-native-safe-area-context

## Quick Start

```bash
npm install
npx expo run:ios   # or npx expo run:android
```

## Project Structure

```
assets/onnx/                    # English ONNX models (for testing)
├── encoder_model_int8.onnx     # ~8MB
├── decoder_model_merged_int8.onnx  # ~20MB
├── tokenizer.json
└── tokenizer_config.json

assets/onnx-ar/                 # Arabic ONNX models (after export)
└── (run export script)

src/
└── MoonshineONNX.ts            # ONNX inference service

scripts/
└── export_moonshine_ar_onnx.py # Arabic model export script
```

## Exporting Arabic Model to ONNX

The English model is included for pipeline validation. To get Arabic transcription:

```bash
# Install Python dependencies
pip install transformers optimum[exporters] onnx onnxruntime

# Export Arabic model
python scripts/export_moonshine_ar_onnx.py --output-dir ./assets/onnx-ar

# With int8 quantization (recommended for mobile)
python scripts/export_moonshine_ar_onnx.py --output-dir ./assets/onnx-ar --quantize int8
```

Then update `App.tsx` to point to the Arabic models:

```typescript
const ONNX_CONFIG = {
  encoderAsset: require('./assets/onnx-ar/encoder_model_int8.onnx'),
  decoderAsset: require('./assets/onnx-ar/decoder_model_merged_int8.onnx'),
  tokenizerAsset: require('./assets/onnx-ar/tokenizer.json'),
};
```

## How It Works

1. **Audio Capture**: Record 16kHz audio with `react-native-audio-api`
2. **Preprocessing**: Normalize audio to zero mean/unit variance
3. **Encoder**: Run audio through ONNX encoder model
4. **Decoder**: Greedy decode with encoder output (autoregressive)
5. **Tokenizer**: Decode BPE tokens to text

Key features:
- No mel spectrogram needed - Moonshine takes raw audio directly
- Uses RoPE (Rotary Position Embedding) for variable-length audio
- Optimized for real-time transcription on edge devices

## Model Sources

- English: [onnx-community/moonshine-tiny-ONNX](https://huggingface.co/onnx-community/moonshine-tiny-ONNX)
- Arabic: [UsefulSensors/moonshine-tiny-ar](https://huggingface.co/UsefulSensors/moonshine-tiny-ar)
- Docs: [Moonshine on HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/moonshine)

## Links

- [Moonshine Paper](https://huggingface.co/papers/2410.15608)
- [ONNX Runtime React Native](https://onnxruntime.ai/docs/tutorials/react-native/)
- [react-native-audio-api](https://github.com/nickapps-dev/react-native-audio-api)
