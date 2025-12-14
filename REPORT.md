# ColQwen3 vLLM Embedding Support

## Summary

This PR enables serving the ColQwen3 multimodal embedding model using vLLM's embedding API. ColQwen3 is a ColPali-style retrieval model based on Qwen3-VL that produces 320-dimensional normalized multi-vector embeddings.

### Model Specifications

| Feature | Detail |
|---------|--------|
| Architecture | Qwen3-VL 4B (Encoder-only variant) + 320-dim Projection Head |
| Methodology | ColPali-style Late Interaction (MaxSim scoring) |
| Token Budget | Up to 1,280 visual tokens per page |
| Output | Multi-vector (Seq_Len × 320), L2-normalized |
| Precision | bfloat16 weights, FlashAttention 2 enabled |

## Current Status

| Mode | Status | Notes |
|------|--------|-------|
| Text Embeddings | ✅ Working | Fully functional via `llm.embed()` |
| Image Embeddings | ⚠️ Partial | Works via HuggingFace API; vLLM multimodal profiling times out |

## Text Embedding Usage (Working)

```python
from vllm import LLM

llm = LLM(
    model="./tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512",
    task="embed",
    trust_remote_code=True,
    hf_overrides={"architectures": ["TransformersMultiModalEmbeddingModel"]},
    limit_mm_per_prompt={"image": 0, "video": 0},  # Text-only mode
    enforce_eager=True,
    max_model_len=2048,
    dtype="bfloat16",
)

outputs = llm.embed(["Retrieve the city of Singapore"])
# Returns 320-dimensional normalized embeddings
```

## Image Embedding via HuggingFace (Working)

For image embeddings, use the HuggingFace Transformers API directly:

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "TomoroAI/tomoro-colqwen3-embed-4b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Image embedding
image = Image.open("document.png").convert("RGB")
inputs = processor.process_images(images=[image])
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model(**inputs)
embeddings = outputs.embeddings  # Shape: (batch, seq_len, 320)

# Text embedding
inputs = processor.process_texts(texts=["Retrieve the city of Singapore"])
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model(**inputs)
embeddings = outputs.embeddings

# Scoring (MaxSim)
scores = processor.score_multi_vector(query_embeddings, doc_embeddings)
```

## Changes Made

### `modeling_colqwen3.py`

1. **Added `get_rope_index` method**: Delegates M-RoPE position computation to the underlying `Qwen3VLModel`. Includes guards for empty `image_grid_thw`/`video_grid_thw` tensors.

2. **Added `get_image_features` method**: Required for vLLM's `TransformersMultiModalEmbeddingModel` wrapper. Extracts vision embeddings from pixel values via the Qwen3VL vision encoder.

### `processing_colqwen3.py`

1. **Fixed placeholder token handling**: Added logic to strip `<|image_pad|>` and `<|video_pad|>` tokens from `extra_text` when images/videos are provided. This handles vLLM's multimodal input format which passes placeholders in the text.

### `main.py`

- Configured vLLM for text-only embeddings
- Added documentation for image embedding alternatives

## Failed Approaches

### 1. `TransformersEmbeddingModel` + `dtype="float16"`
**Error**: `RuntimeError: expected scalar type Half but found BFloat16`  
**Cause**: Model weights are bf16 but vLLM tried to run in fp16.

### 2. `TransformersEmbeddingModel` + `dtype="bfloat16"`
**Error**: `AssertionError: M-RoPE support is not implemented.`  
**Cause**: Generic embedding wrapper doesn't support Qwen3VL's multimodal rotary position embeddings.

### 3. Custom vLLM architecture (`ColQwen3ForRetrieval`)
**Error**: `Model architectures ['ColQwen3ForRetrieval'] are not supported`  
**Cause**: vLLM's model registry requires explicit registration of new architectures.

### 4. `TransformersMultiModalEmbeddingModel` (without `get_rope_index`)
**Error**: `AttributeError: 'ColQwen3' object has no attribute 'get_rope_index'`  
**Cause**: vLLM's multimodal wrapper expects this method for M-RoPE computation.

### 5. Initial `get_rope_index` delegating to `self.vlm.get_rope_index`
**Error**: `AttributeError: 'Qwen3VLForConditionalGeneration' object has no attribute 'get_rope_index'`  
**Cause**: The method exists on `Qwen3VLModel`, not `Qwen3VLForConditionalGeneration`. Fixed by delegating to `self.vlm.model.get_rope_index`.

### 6. Empty tensor handling
**Error**: `IndexError: too many indices for tensor of dimension 1`  
**Cause**: vLLM passes empty tensors for `video_grid_thw` when no videos are present. Fixed by converting empty tensors to `None`.

### 7. vLLM multimodal profiling with images
**Error**: Timeout during memory profiling with `limit_mm_per_prompt={"image": 1}`  
**Cause**: vLLM's profiler creates 6 maximum-size dummy images for memory estimation, which overwhelms the vision encoder. The ColQwen3Processor's `size.longest_edge` was set to 1,310,720 pixels.

### 8. Processor placeholder mismatch
**Error**: `ValueError: Number of image tokens does not match provided images.`  
**Cause**: vLLM passes `text='<|image_pad|>'` with images, but ColQwen3Processor already adds `<|image_pad|>` via `visual_prompt_prefix`, creating duplicate tokens. Fixed by stripping placeholder tokens from `extra_text`.

## Multimodal vLLM Roadmap

To enable full multimodal support in vLLM, the following would be needed:

1. **Custom profiling**: Override vLLM's dummy image generation to use smaller images (e.g., 512×512) that match the model's typical usage.

2. **Custom processor registration**: Implement vLLM's `BaseMultiModalProcessor` interface with:
   - `get_supported_mm_limits()` returning `{"image": 1, "video": 0}`
   - `_get_mm_fields_config()` mapping `pixel_values` and `image_grid_thw`
   - Proper dummy data builders for memory profiling

3. **Model registration**: Add ColQwen3 to vLLM's model registry as a native multimodal embedding model.

## Testing

```bash
# Text embeddings via vLLM
uv run python main.py

# Image embeddings via HuggingFace
uv run python test_vision.py
```

## Key Implementation Details

1. **Config inheritance**: `ColQwen3Config` extends `Qwen3VLConfig` with `model_type="colqwen3"`
2. **Output compatibility**: `ColQwen3ForRetrievalOutput` inherits from `BaseModelOutput` and exposes `last_hidden_state`
3. **M-RoPE support**: `get_rope_index` delegates to `self.vlm.model.get_rope_index` with empty tensor guards
4. **Vision features**: `get_image_features` extracts embeddings from `self.vlm.model.visual`
5. **Forward signature**: Accepts `pixel_values`, `image_grid_thw`, and `**kwargs` for vLLM compatibility
