# ColQwen3 vLLM Embedding Support

## Summary

This PR enables serving the ColQwen3 multimodal embedding model using vLLM's embedding API. ColQwen3 is a retrieval model based on Qwen3-VL that produces 320-dimensional normalized embeddings.

## Changes

### `modeling_colqwen3.py`

- **Added `get_rope_index` method**: Delegates M-RoPE position computation to the underlying `Qwen3VLModel`. Includes guards for empty `image_grid_thw`/`video_grid_thw` tensors that vLLM passes for text-only inputs.

### `main.py`

- Configured vLLM to use `TransformersMultiModalEmbeddingModel` architecture wrapper
- Set `dtype="bfloat16"` to match model weights
- Set `limit_mm_per_prompt={"image": 0, "video": 0}` for text-only embeddings
- Enabled `enforce_eager=True` (required for Transformers wrapper)

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

## Working Solution

```python
llm = LLM(
    model="./tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-512",
    task="embed",
    trust_remote_code=True,
    hf_overrides={"architectures": ["TransformersMultiModalEmbeddingModel"]},
    limit_mm_per_prompt={"image": 0, "video": 0},
    enforce_eager=True,
    max_model_len=2048,
    dtype="bfloat16",
)

outputs = llm.embed(["Hello, my name is"])
# Returns 320-dimensional normalized embeddings
```

## Key Implementation Details

1. **Config inheritance**: `ColQwen3Config` extends `Qwen3VLConfig` with `model_type="colqwen3"`
2. **Output compatibility**: `ColQwen3ForRetrievalOutput` inherits from `BaseModelOutput` and exposes `last_hidden_state`
3. **M-RoPE support**: `get_rope_index` delegates to `self.vlm.model.get_rope_index` with empty tensor guards
4. **Forward signature**: Accepts `**kwargs` to handle extra vLLM arguments

## Testing

```bash
uv run python main.py
# Output: Prompt: 'Hello, my name is' | Embeddings: [...] (size=320)
```
