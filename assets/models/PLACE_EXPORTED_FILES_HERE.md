# Moonshine Arabic Model Files

Place the exported ExecuTorch files here:

```
assets/models/
├── moonshine_tiny_ar_encoder_xnnpack.pte   (~50-100MB)
├── moonshine_tiny_ar_decoder_xnnpack.pte   (~50-100MB)
└── tokenizer.json                          (~1MB)
```

## Export Command (for maintainer)

```python
# Export script reference - see MAINTAINER_REQUEST.md
from transformers import MoonshineForConditionalGeneration
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny-ar")

# Export encoder
encoder = model.get_encoder()
# ... export to moonshine_tiny_ar_encoder_xnnpack.pte

# Export decoder
decoder = model.get_decoder()
# ... export to moonshine_tiny_ar_decoder_xnnpack.pte

# Copy tokenizer.json from UsefulSensors/moonshine-tiny-ar
```
