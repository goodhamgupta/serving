# ColQwen3 8B Embedding Model Benchmark Report

## Overview

This report compares the performance and memory consumption of the 8B ColQwen3 embedding model variants:

| Model | Description | Model ID |
|-------|-------------|----------|
| **BASE** | Original BF16 model | `TomoroAI/tomoro-colqwen3-embed-8b` |
| **Quantized** | AutoRound W4A16 | `shubhamg2208/tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024` |

## Important Notes

> **The AutoRound quantization was applied only to the text tower (language model). The vision tower remains unchanged in FP16/BF16.** This means memory improvements are observed in the text encoding workloads.

> ⚠️ **Backend Limitation**: The optimal quantization backend (`gptqmodel>=2.0`) could not be installed due to compatibility issues. The benchmarks use a fallback backend which does not provide the expected throughput improvements. With the proper backend installed, the quantized model should show **improved throughput** over the base model.

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| CUDA Version | 12.8 |
| Attention Implementation | SDPA |
| Text Samples | 32 |
| Text Batch Size | 8 |
| Warmup Steps | 3 |
| Measurement Steps | 10 |

---

## Summary Comparison (Text Encoding Only)

### Memory Usage

| Metric | BASE | Quantized | Reduction |
|--------|------|-----------|-----------|
| **Peak Memory (MB)** | 16,791 | 6,998 | **-58.3%** |
| **Allocated Memory (MB)** | 16,754 | 6,962 | **-58.4%** |

### Throughput

| Metric | BASE | Quantized | Note |
|--------|------|-----------|------|
| **Throughput (items/s)** | 182.6 | 105.9 | *Fallback backend - see note above* |
| **Latency (ms/batch)** | 43.82 | 75.58 | *Fallback backend - see note above* |

---

## Detailed Results

### Text Queries Benchmark

| Metric | BASE | Quantized |
|--------|------|-----------|
| Total Items | 80 | 80 |
| Total Time (s) | 0.438 | 0.756 |
| Throughput (items/s) | 182.6 | 105.9 |
| Mean Latency (ms/batch) | 43.82 | 75.58 |
| Std Latency (ms/batch) | 5.26 | 2.27 |
| CUDA Allocated (MB) | 16,754 | 6,962 |
| CUDA Peak Allocated (MB) | 16,791 | 6,998 |
| CUDA Peak Reserved (MB) | 16,818 | 7,304 |

---

## Key Findings

### 1. Memory Efficiency ✅
The quantized 8B model achieves **~58% memory reduction**:
- Peak memory: 16,791 MB → 6,998 MB
- This enables deployment on GPUs with 8-16GB VRAM instead of requiring 24GB+

### 2. Throughput (Backend Dependent) ⚠️
Current results show slower throughput due to missing optimized backend.

**To enable optimal performance**, install:
```bash
pip install -v "gptqmodel>=2.0" --no-build-isolation
pip install 'numpy<2.0'
```

With the proper backend, W4A16 quantization typically provides:
- **1.5-2x throughput improvement** over FP16/BF16
- Lower latency due to reduced memory bandwidth requirements

---

## Comparison with 4B Models

| Model | BASE Memory | Quantized Memory | Memory Reduction |
|-------|-------------|------------------|------------------|
| **4B** | ~8.5 GB | ~3.5 GB | ~59% |
| **8B** | ~16.8 GB | ~7.0 GB | ~58% |

Both model sizes show consistent memory reduction from quantization.

---

## Recommendations

1. **Use Quantized 8B model when**:
   - GPU memory is limited (8-16GB GPUs)
   - Running on consumer hardware (RTX 3090, 4090)
   - Need larger model capacity with memory constraints

2. **Install optimized backend** for production:
   ```bash
   pip install -v "gptqmodel>=2.0" --no-build-isolation
   ```

---

## Reproducibility

```bash
# Benchmark BASE 8B model (text only)
uv run python benchmark_8b.py --only_base --text_only --text_samples 32 \
    --text_batch_size 8 --output_json base_8b_results.json

# Benchmark Quantized 8B model (text only)
uv run python benchmark_8b.py --only_awq --text_only --text_samples 32 \
    --text_batch_size 8 --output_json awq_8b_results.json
```

---

*Report generated on NVIDIA A100-SXM4-40GB with CUDA 12.8*
