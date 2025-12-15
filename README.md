# Moonshine Arabic - Live Transcription

On-device Arabic speech-to-text using Moonshine-tiny-ar with React Native ExecuTorch.

## Status: Waiting for Model Export

App is ready. Needs `moonshine-tiny-ar` exported to ExecuTorch .pte format.

## Why Moonshine Arabic?

| Model | Params | Arabic WER |
|-------|--------|------------|
| **moonshine-tiny-ar** | 27M | **20.76** |
| whisper-tiny | 39M | 66.01 |

3x more accurate than Whisper Tiny.

## Stack

- Expo SDK 54 / RN 0.81 / New Architecture
- react-native-executorch 0.6.0
- react-native-audio-api (16kHz capture)

## Run

```bash
npm install
npx expo run:ios   # or android
```

## Files Needed

```
assets/models/
├── moonshine_tiny_ar_encoder_xnnpack.pte
├── moonshine_tiny_ar_decoder_xnnpack.pte
└── tokenizer.json
```

Or hosted at: `software-mansion/react-native-executorch-moonshine-tiny-ar`

## Test with Whisper

Edit `App.tsx`:
```typescript
import { WHISPER_TINY_EN } from 'react-native-executorch';
// Change model: MOONSHINE_ARABIC_MODEL → WHISPER_TINY_EN
```

## Links

- [moonshine-tiny-ar](https://huggingface.co/UsefulSensors/moonshine-tiny-ar)
- [react-native-executorch](https://docs.swmansion.com/react-native-executorch/)
