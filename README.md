# ColQwen3 Embedding Model Benchmarks

Benchmark suite for comparing ColQwen3 embedding models (BASE vs AWQ quantized).

## Models

| Size | BASE Model | AWQ Model |
|------|------------|-----------|
| 4B | `TomoroAI/tomoro-colqwen3-embed-4b` | `shubhamg2208/tomoro-ai-colqwen3-embed-4b-w4a16-autoawq-seqlen-1024` |
| 8B | `TomoroAI/tomoro-colqwen3-embed-8b` | `shubhamg2208/tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024` |

## Requirements

- Python 3.11+
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

## Running Benchmarks

### 4B Models

```bash
# Run both BASE and AWQ
uv run python benchmark.py

# Run BASE model only
uv run python benchmark.py --only_base --output_json base_results.json

# Run AWQ model only
uv run python benchmark.py --only_awq --output_json awq_results.json
```

### 8B Models

```bash
# Run both BASE and AWQ
uv run python benchmark_8b.py

# Run BASE model only
uv run python benchmark_8b.py --only_base --output_json base_8b_results.json

# Run AWQ model only
uv run python benchmark_8b.py --only_awq --output_json awq_8b_results.json
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--text_samples` | 64 | Number of text samples |
| `--text_batch_size` | 8 | Batch size for text |
| `--image_samples` | 16 | Number of image samples |
| `--image_batch_size` | 4 | Batch size for images |
| `--image_size` | 512 | Image dimensions (pixels) |
| `--warmup_steps` | 3 | Warmup iterations |
| `--measure_steps` | 10 | Measurement iterations |
| `--only_base` | - | Benchmark BASE model only |
| `--only_awq` | - | Benchmark AWQ model only |
| `--output_json` | - | Save results to JSON file |

### Example with Custom Parameters

```bash
uv run python benchmark.py \
    --text_samples 32 \
    --text_batch_size 8 \
    --image_samples 16 \
    --image_batch_size 4 \
    --warmup_steps 5 \
    --measure_steps 20 \
    --output_json results.json
```

## Reports

- [REPORT.md](REPORT.md) - 4B model benchmark results
- [REPORT_8B.md](REPORT_8B.md) - 8B model benchmark results

## Notes

- AWQ quantization is applied only to the text tower; the vision tower remains in FP16/BF16
- Run models separately (`--only_base` / `--only_awq`) to avoid GPU memory conflicts
- Results are saved to JSON for further analysis when using `--output_json`
