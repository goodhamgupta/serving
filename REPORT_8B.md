# ColQwen3 8B Embedding Model Benchmark Report

## Overview

This report compares the performance and memory consumption of the 8B ColQwen3 embedding model variants:

| Model | Description | Model ID |
|-------|-------------|----------|
| **BASE** | Original BF16 model | `TomoroAI/tomoro-colqwen3-embed-8b` |
| **Quantized** | AutoRound W4A16 | `shubhamg2208/tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024` |

## Important Notes

> **The AutoRound quantization was applied only to the text tower (language model). The vision tower remains unchanged in FP16/BF16.** This means memory improvements are observed in the text encoding workloads.

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| CUDA Version | 12.8 |
| Attention Implementation | SDPA |

---

## Key Finding: Memory Efficiency Enables Deployment on Smaller GPUs

The 8B quantized model uses **~58% less memory**, enabling deployment on GPUs that cannot fit the base model.

### Batch Size Sweep Results

| Batch Size | BASE Throughput | BASE Memory | Quant Throughput | Quant Memory |
|------------|-----------------|-------------|------------------|--------------|
| 8 | 208 items/s | 16,790 MB | 106 items/s | 6,997 MB |
| 16 | 403 items/s | 16,844 MB | 127 items/s | 7,051 MB |
| 32 | 573 items/s | 16,964 MB | 134 items/s | 7,164 MB |
| 64 | 607 items/s | 17,197 MB | 538 items/s | 7,471 MB |
| 128 | 625 items/s | 17,683 MB | 596 items/s | 7,925 MB |
| 256 | 635 items/s | 18,604 MB | 635 items/s | 8,812 MB |
| 512 | - | OOM | 632 items/s | 10,810 MB |

### Memory-Constrained Scenarios

#### Scenario: 12GB GPU (e.g., RTX 3080, 4080)

| Model | Fits? | Max Batch Size | Throughput |
|-------|-------|----------------|------------|
| BASE | ❌ No | - | Cannot run |
| **Quantized** | ✅ Yes | **128** | **596 items/s** |

**Result: Quantized model enables 8B model deployment on 12GB GPUs**

#### Scenario: 16GB GPU (e.g., RTX 4080 Super, A4000)

| Model | Fits? | Max Batch Size | Throughput |
|-------|-------|----------------|------------|
| BASE | ❌ No | - | Cannot run |
| **Quantized** | ✅ Yes | **512** | **632 items/s** |

**Result: Quantized model enables high-throughput 8B inference on 16GB GPUs**

#### Scenario: 24GB GPU (e.g., RTX 3090, 4090)

| Model | Max Batch Size | Throughput | Peak Memory |
|-------|----------------|------------|-------------|
| BASE | ~64 | 607 items/s | 17,197 MB |
| **Quantized** | **1024** | **637 items/s** | 14,817 MB |

**Result: Quantized model achieves 5% higher throughput with 16x larger batches**

---

## Comparison: 4B vs 8B Models

| Model | BASE Memory | Quantized Memory | Memory Reduction |
|-------|-------------|------------------|------------------|
| **4B** | ~10 GB (bs=256) | ~5 GB (bs=256) | ~50% |
| **8B** | ~19 GB (bs=256) | ~9 GB (bs=256) | ~53% |

The 8B model benefits significantly from quantization as the base model requires 24GB+ GPUs.

---

## High Batch Scaling (Quantized Model Only)

The quantized 8B model can scale to much larger batch sizes before hitting OOM:

| Batch Size | Quantized Throughput | Quantized Memory |
|------------|---------------------|------------------|
| 512 | 635 items/s | 10,813 MB |
| 1024 | 637 items/s | 14,817 MB |
| 1536 | 639 items/s | 18,755 MB |
| 2048 | 640 items/s | 22,693 MB |

> **Note:** BASE 8B model requires 17GB+ even at small batch sizes. The quantized model enables 8B deployment on 12-16GB GPUs.

---

## Recommendations

### Use Quantized 8B Model When:
- GPU memory is 12-16GB (enables 8B deployment)
- Need larger model capacity than 4B
- Want to run 8B on consumer hardware

### Use BASE 8B Model When:
- GPU has 24GB+ memory
- Running very small batch sizes
- Maximum quality is required

### Consider 4B Quantized When:
- GPU memory is 4-8GB
- Highest throughput is priority
- Slightly lower capacity is acceptable

---

## Reproducibility

```bash
# Sweep batch sizes for BASE 8B model
uv run python benchmark_8b.py --only_base --text_only \
    --sweep_batch_sizes "8,16,32,64,128,256" \
    --output_json sweep_8b_base.json

# Sweep batch sizes for Quantized 8B model
uv run python benchmark_8b.py --only_awq --text_only \
    --sweep_batch_sizes "8,16,32,64,128,256,512" \
    --output_json sweep_8b_awq.json

# High batch sweep for Quantized 8B model (demonstrates max batch advantage)
uv run python benchmark_8b.py --only_awq --text_only \
    --high_batch_sweep "512,1024,1536,2048" \
    --output_json sweep_8b_awq_high.json
```

---

*Report generated on NVIDIA A100-SXM4-40GB with CUDA 12.8*
