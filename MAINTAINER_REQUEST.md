# Export Request: moonshine-tiny-ar → ExecuTorch

## Source Model
`UsefulSensors/moonshine-tiny-ar` (27M params, 20.76 WER - 3x better than Whisper tiny)

## Files Needed

```
software-mansion/react-native-executorch-moonshine-tiny-ar/
├── xnnpack/
│   ├── moonshine_tiny_ar_encoder_xnnpack.pte
│   └── moonshine_tiny_ar_decoder_xnnpack.pte
└── tokenizer.json
```

## Export Reference

Similar to Whisper export structure:
```python
# Pseudo-code based on existing Whisper export pattern
from transformers import MoonshineForConditionalGeneration
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny-ar")

# Export encoder
encoder_program = to_edge_transform_and_lower(
    torch.export.export(model.get_encoder(), sample_audio),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

# Export decoder
decoder_program = to_edge_transform_and_lower(
    torch.export.export(model.get_decoder(), sample_tokens),
    partitioner=[XnnpackPartitioner()]
).to_executorch()
```

## Test App Ready

Repo: https://github.com/baiamansama/moonshine.rn

- Expo SDK 54, RN 0.81, react-native-executorch 0.6.0
- Audio capture at 16kHz working
- Just needs the .pte files to test

Once exported, I'll update URLs and test immediately.
