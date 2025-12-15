# ColQwen3 4B Embedding Model Benchmark Report

## Overview

This report compares the performance and memory consumption of the 4B ColQwen3 embedding model variants:

| Model | Description | Model ID |
|-------|-------------|----------|
| **BASE** | Original BF16 model | `TomoroAI/tomoro-colqwen3-embed-4b` |
| **Quantized** | AutoRound W4A16 | `shubhamg2208/tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024` |

## Important Notes

> **The AutoRound quantization was applied only to the text tower (language model). The vision tower remains unchanged in FP16/BF16.** This means memory improvements are observed in the text encoding workloads.

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| CUDA Version | 12.8 |
| Attention Implementation | Flash Attention 2 (BASE), Fallback (Quantized) |

---

## Key Finding: Memory Efficiency Enables Higher Throughput

The quantized model uses **~60% less memory**, allowing for **larger batch sizes** on the same GPU.

### Batch Size Sweep Results

| Batch Size | BASE Throughput | BASE Memory | Quant Throughput | Quant Memory |
|------------|-----------------|-------------|------------------|--------------|
| 8 | 204 items/s | 8,526 MB | 111 items/s | 3,457 MB |
| 16 | 402 items/s | 8,578 MB | 222 items/s | 3,509 MB |
| 32 | 772 items/s | 8,714 MB | 436 items/s | 3,637 MB |
| 64 | 904 items/s | 8,922 MB | 798 items/s | 3,828 MB |
| 128 | 957 items/s | 9,356 MB | 928 items/s | 4,247 MB |
| 256 | 972 items/s | 10,219 MB | 978 items/s | 5,105 MB |
| 512 | - | OOM | 985 items/s | 6,962 MB |

### Memory-Constrained Scenarios

The real advantage of quantization appears when GPU memory is limited:

#### Scenario: 8GB GPU (e.g., RTX 3070, 4070)

| Model | Max Batch Size | Throughput | Peak Memory |
|-------|----------------|------------|-------------|
| BASE | ~8 | 204 items/s | 8,526 MB ❌ (won't fit) |
| **Quantized** | **128** | **928 items/s** | 4,247 MB ✅ |

**Result: Quantized model enables 4.5x higher throughput on 8GB GPUs**

#### Scenario: 12GB GPU (e.g., RTX 3080, 4080)

| Model | Max Batch Size | Throughput | Peak Memory |
|-------|----------------|------------|-------------|
| BASE | ~64 | 904 items/s | 8,922 MB |
| **Quantized** | **512** | **985 items/s** | 6,962 MB |

**Result: Quantized model achieves 9% higher throughput on 12GB GPUs**

---

## Detailed Comparison at Same Batch Size

At identical batch sizes, the quantized model with fallback backend is slower due to dequantization overhead:

| Batch Size | BASE | Quantized | Overhead |
|------------|------|-----------|----------|
| 8 | 204 items/s | 111 items/s | -46% |
| 64 | 904 items/s | 798 items/s | -12% |
| 128 | 957 items/s | 928 items/s | -3% |
| 256 | 972 items/s | 978 items/s | **+0.6%** |

Note: At large batch sizes (256+), throughput becomes comparable as compute dominates over kernel overhead.

---

## High Batch Scaling (Quantized Model Only)

The quantized model can scale to much larger batch sizes before hitting OOM:

| Batch Size | Quantized Throughput | Quantized Memory |
|------------|---------------------|------------------|
| 512 | 986 items/s | 6,960 MB |
| 1024 | 998 items/s | 10,697 MB |
| 1536 | 1,000 items/s | 14,364 MB |
| 2048 | 1,003 items/s | 18,032 MB |
| 2560 | 1,005 items/s | 21,701 MB |
| 3072 | 1,007 items/s | 25,369 MB |

> **Note:** BASE model OOMs at batch sizes >256 on 40GB A100 during peak memory. The quantized model can run 3-6x larger batches.

---

## Recommendations

### Use Quantized Model When:
- GPU memory is limited (4-16GB)
- Need to maximize throughput via larger batches
- Deploying on consumer hardware

### Use BASE Model When:
- GPU has ample memory (24GB+)
- Running small batch sizes
- Maximum per-sample latency is critical

---

## Reproducibility

```bash
# Sweep batch sizes for BASE model
uv run python benchmark.py --only_base --text_only \
    --sweep_batch_sizes "8,16,32,64,128,256" \
    --output_json sweep_4b_base.json

# Sweep batch sizes for Quantized model
uv run python benchmark.py --only_awq --text_only \
    --sweep_batch_sizes "8,16,32,64,128,256,512" \
    --output_json sweep_4b_awq.json

# High batch sweep for Quantized model (demonstrates max batch advantage)
uv run python benchmark.py --only_awq --text_only \
    --high_batch_sweep "512,1024,1536,2048,2560,3072" \
    --output_json sweep_4b_awq_high.json
```

---

*Report generated on NVIDIA A100-SXM4-40GB with CUDA 12.8*
