#!/usr/bin/env python3
"""
Benchmark throughput and memory consumption for ColQwen3 embedding models.
Compares the original BF16 model vs AWQ quantized (W4A16) version.
"""

import argparse
import gc
import json
import statistics
import time
from typing import Dict, List

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BASE_MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-8b"
AWQ_MODEL_ID = "shubhamg2208/tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

DEFAULT_TEXT_SAMPLES = 64
DEFAULT_TEXT_BATCH_SIZE = 8
DEFAULT_IMAGE_SAMPLES = 16
DEFAULT_IMAGE_BATCH_SIZE = 4
DEFAULT_IMAGE_SIZE = 512
DEFAULT_WARMUP_STEPS = 3
DEFAULT_MEASURE_STEPS = 10


def _human_mb(bytes_val: int) -> float:
    return bytes_val / (1024 ** 2)


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_text_batches(
    processor,
    num_samples: int,
    batch_size: int,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    """Prepare tokenized text batches using processor.process_texts."""
    texts = [f"Retrieve information about topic number {i}" for i in range(num_samples)]
    
    batches = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_texts = texts[start:end]
        batch = processor.process_texts(texts=batch_texts)
        batch = {k: v.to(device) for k, v in batch.items()}
        batches.append(batch)
    
    return batches


def prepare_image_batches(
    processor,
    num_samples: int,
    batch_size: int,
    image_size: int,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    """Prepare image batches using processor.process_images with synthetic images."""
    images = []
    for _ in range(num_samples):
        arr = (np.random.rand(image_size, image_size, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        images.append(img)
    
    batches = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_imgs = images[start:end]
        features = processor.process_images(images=batch_imgs)
        features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        batches.append(features)
    
    return batches


def run_benchmark(
    model,
    batches: List[Dict[str, torch.Tensor]],
    warmup_steps: int,
    measure_steps: int,
    scenario_name: str,
) -> Dict[str, float]:
    """Run warmup + timed inference and return timing and memory statistics."""
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    
    model.eval()
    with torch.inference_mode():
        for step in range(min(warmup_steps, len(batches))):
            _ = model(**batches[step % len(batches)])
            _synchronize()
    
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    
    per_batch_times: List[float] = []
    total_items = 0
    
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=DTYPE):
            steps = 0
            while steps < measure_steps:
                batch = batches[steps % len(batches)]
                
                bs = None
                for key in ["input_ids", "pixel_values"]:
                    if key in batch:
                        bs = batch[key].shape[0]
                        break
                if bs is None:
                    first_key = next(iter(batch.keys()))
                    bs = batch[first_key].shape[0] if hasattr(batch[first_key], 'shape') else 1
                
                start = time.perf_counter()
                _ = model(**batch)
                _synchronize()
                end = time.perf_counter()
                
                per_batch_times.append(end - start)
                total_items += bs
                steps += 1
    
    _synchronize()
    if use_cuda:
        current_alloc = torch.cuda.memory_allocated(device)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        current_alloc = peak_alloc = peak_reserved = 0
    
    total_time = sum(per_batch_times)
    per_batch_ms = [t * 1000.0 for t in per_batch_times]
    mean_latency_ms = statistics.mean(per_batch_ms)
    std_latency_ms = statistics.pstdev(per_batch_ms) if len(per_batch_ms) > 1 else 0.0
    throughput = total_items / total_time
    
    return {
        "scenario": scenario_name,
        "total_items": total_items,
        "total_time_s": total_time,
        "throughput_items_per_s": throughput,
        "latency_mean_ms_per_batch": mean_latency_ms,
        "latency_std_ms_per_batch": std_latency_ms,
        "cuda_memory_allocated_MB": _human_mb(current_alloc),
        "cuda_max_memory_allocated_MB": _human_mb(peak_alloc),
        "cuda_max_memory_reserved_MB": _human_mb(peak_reserved),
    }


def print_results(prefix: str, stats: Dict[str, float]):
    print(f"\n{'='*60}")
    print(f"{prefix} | {stats['scenario']}")
    print(f"{'='*60}")
    print(f"Total items                : {stats['total_items']}")
    print(f"Total time (s)             : {stats['total_time_s']:.3f}")
    print(f"Throughput (items/s)       : {stats['throughput_items_per_s']:.2f}")
    print(f"Mean latency (ms / batch)  : {stats['latency_mean_ms_per_batch']:.3f}")
    print(f"Std latency (ms / batch)   : {stats['latency_std_ms_per_batch']:.3f}")
    print(f"CUDA allocated (MB)        : {stats['cuda_memory_allocated_MB']:.1f}")
    print(f"CUDA peak allocated (MB)   : {stats['cuda_max_memory_allocated_MB']:.1f}")
    print(f"CUDA peak reserved (MB)    : {stats['cuda_max_memory_reserved_MB']:.1f}")


def load_base_model(processor_max_tokens: int = 1280):
    """Load the original BF16 ColQwen3 embedding model."""
    print(f"\nLoading base model: {BASE_MODEL_ID}")
    
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        max_num_visual_tokens=processor_max_tokens,
    )
    
    model = AutoModel.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=DEVICE,
    ).eval()
    
    return model, processor


def load_awq_model(processor_max_tokens: int = 1280):
    """Load the AutoRound W4A16 quantized ColQwen3 embedding model."""
    print(f"\nLoading quantized model (AutoRound W4A16): {AWQ_MODEL_ID}")
    
    processor = AutoProcessor.from_pretrained(
        AWQ_MODEL_ID,
        trust_remote_code=True,
        max_num_visual_tokens=processor_max_tokens,
    )
    
    model = AutoModel.from_pretrained(
        AWQ_MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    
    return model, processor


def cleanup_model(model):
    """Cleanup model between runs."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sweep_text_batch_sizes(
    model,
    processor,
    batch_sizes: List[int],
    num_samples: int,
    warmup_steps: int,
    measure_steps: int,
    scenario_prefix: str,
) -> List[tuple]:
    """Sweep through batch sizes to find max throughput under memory constraints."""
    results = []
    for bs in batch_sizes:
        print(f"\n[{scenario_prefix}] Trying batch size = {bs}")
        try:
            text_batches = prepare_text_batches(
                processor,
                num_samples=max(num_samples, bs * 2),
                batch_size=bs,
                device=DEVICE,
            )
            stats = run_benchmark(
                model,
                text_batches,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                scenario_name=f"text_queries_bs{bs}",
            )
            print_results(scenario_prefix, stats)
            results.append((bs, stats))
            del text_batches
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[{scenario_prefix}] OOM at batch size = {bs}, stopping sweep.")
                gc.collect()
                torch.cuda.empty_cache()
                break
            else:
                raise
    return results


def print_sweep_summary(base_results: List[tuple], awq_results: List[tuple]):
    """Print comparison of max batch size throughput."""
    print("\n" + "="*70)
    print("MAX BATCH SIZE THROUGHPUT COMPARISON")
    print("="*70)
    
    if base_results:
        base_max_bs, base_max_stats = base_results[-1]
        print(f"\nBASE Model:")
        print(f"  Max batch size achieved: {base_max_bs}")
        print(f"  Throughput at max bs: {base_max_stats['throughput_items_per_s']:.2f} items/s")
        print(f"  Peak memory: {base_max_stats['cuda_max_memory_allocated_MB']:.1f} MB")
    
    if awq_results:
        awq_max_bs, awq_max_stats = awq_results[-1]
        print(f"\nQuantized Model:")
        print(f"  Max batch size achieved: {awq_max_bs}")
        print(f"  Throughput at max bs: {awq_max_stats['throughput_items_per_s']:.2f} items/s")
        print(f"  Peak memory: {awq_max_stats['cuda_max_memory_allocated_MB']:.1f} MB")
    
    if base_results and awq_results:
        base_max_bs, base_max_stats = base_results[-1]
        awq_max_bs, awq_max_stats = awq_results[-1]
        
        throughput_ratio = awq_max_stats['throughput_items_per_s'] / base_max_stats['throughput_items_per_s']
        batch_ratio = awq_max_bs / base_max_bs
        
        print(f"\nComparison:")
        print(f"  Batch size increase: {batch_ratio:.1f}x ({base_max_bs} → {awq_max_bs})")
        print(f"  Throughput ratio: {throughput_ratio:.2f}x")
        if throughput_ratio > 1:
            print(f"  ✅ Quantized model achieves {(throughput_ratio-1)*100:.1f}% higher throughput at max batch size")
        else:
            print(f"  Quantized model achieves {throughput_ratio*100:.1f}% of base throughput at max batch size")


def print_summary(results: Dict[str, Dict[str, float]]):
    """Print a comparison summary table."""
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<35} {'BASE':<15} {'AWQ':<15} {'Speedup':<10}")
    print("-"*70)
    
    for scenario in ["text_queries", "image_documents"]:
        base_key = f"BASE_{scenario}"
        awq_key = f"AWQ_{scenario}"
        
        if base_key in results and awq_key in results:
            base = results[base_key]
            awq = results[awq_key]
            
            print(f"\n{scenario.upper()}")
            
            speedup = awq["throughput_items_per_s"] / base["throughput_items_per_s"]
            print(f"  {'Throughput (items/s)':<33} {base['throughput_items_per_s']:<15.2f} {awq['throughput_items_per_s']:<15.2f} {speedup:.2f}x")
            
            print(f"  {'Mean latency (ms/batch)':<33} {base['latency_mean_ms_per_batch']:<15.3f} {awq['latency_mean_ms_per_batch']:<15.3f}")
            
            mem_reduction = (base["cuda_max_memory_allocated_MB"] - awq["cuda_max_memory_allocated_MB"]) / base["cuda_max_memory_allocated_MB"] * 100
            print(f"  {'Peak memory (MB)':<33} {base['cuda_max_memory_allocated_MB']:<15.1f} {awq['cuda_max_memory_allocated_MB']:<15.1f} {mem_reduction:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ColQwen3 vs AWQ quantized model")
    parser.add_argument("--text_samples", type=int, default=DEFAULT_TEXT_SAMPLES)
    parser.add_argument("--text_batch_size", type=int, default=DEFAULT_TEXT_BATCH_SIZE)
    parser.add_argument("--image_samples", type=int, default=DEFAULT_IMAGE_SAMPLES)
    parser.add_argument("--image_batch_size", type=int, default=DEFAULT_IMAGE_BATCH_SIZE)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measure_steps", type=int, default=DEFAULT_MEASURE_STEPS)
    parser.add_argument("--only_base", action="store_true", help="Only benchmark base model")
    parser.add_argument("--only_awq", action="store_true", help="Only benchmark AWQ model")
    parser.add_argument("--text_only", action="store_true", help="Only benchmark text (skip images)")
    parser.add_argument("--sweep_batch_sizes", type=str, default=None,
                        help="Comma-separated batch sizes to sweep, e.g. '4,8,16,32,64'")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results as JSON")
    args = parser.parse_args()
    
    batch_sizes = None
    if args.sweep_batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.sweep_batch_sizes.split(",")]
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    all_results = {}
    base_sweep_results = []
    awq_sweep_results = []
    
    if not args.only_awq:
        base_model, base_processor = load_base_model()
        
        if batch_sizes:
            print("\n" + "="*60)
            print("BATCH SIZE SWEEP - BASE MODEL")
            print("="*60)
            base_sweep_results = sweep_text_batch_sizes(
                base_model,
                base_processor,
                batch_sizes=batch_sizes,
                num_samples=args.text_samples,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                scenario_prefix="BASE",
            )
            for bs, stats in base_sweep_results:
                all_results[f"BASE_text_queries_bs{bs}"] = stats
        else:
            print("\nPreparing text batches for BASE model...")
            text_batches = prepare_text_batches(
                base_processor,
                num_samples=args.text_samples,
                batch_size=args.text_batch_size,
                device=DEVICE,
            )
            
            print("\nRunning text benchmark for BASE model...")
            base_text_stats = run_benchmark(
                base_model,
                text_batches,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                scenario_name="text_queries",
            )
            print_results("BASE", base_text_stats)
            all_results["BASE_text_queries"] = base_text_stats
            
            del text_batches
            gc.collect()
            torch.cuda.empty_cache()
        
        if not args.text_only and not batch_sizes:
            print("Preparing image batches for BASE model...")
            image_batches = prepare_image_batches(
                base_processor,
                num_samples=args.image_samples,
                batch_size=args.image_batch_size,
                image_size=args.image_size,
                device=DEVICE,
            )
            
            print("\nRunning image benchmark for BASE model...")
            base_image_stats = run_benchmark(
                base_model,
                image_batches,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                scenario_name="image_documents",
            )
            print_results("BASE", base_image_stats)
            all_results["BASE_image_documents"] = base_image_stats
            
            del image_batches
            gc.collect()
            torch.cuda.empty_cache()
        
        cleanup_model(base_model)
        del base_processor
        gc.collect()
        torch.cuda.empty_cache()
    
    if not args.only_base:
        awq_model, awq_processor = load_awq_model()
        
        if batch_sizes:
            print("\n" + "="*60)
            print("BATCH SIZE SWEEP - QUANTIZED MODEL")
            print("="*60)
            awq_sweep_results = sweep_text_batch_sizes(
                awq_model,
                awq_processor,
                batch_sizes=batch_sizes,
                num_samples=args.text_samples,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                scenario_prefix="QUANT",
            )
            for bs, stats in awq_sweep_results:
                all_results[f"AWQ_text_queries_bs{bs}"] = stats
        else:
            print("\nPreparing text batches for AWQ model...")
            text_batches = prepare_text_batches(
                awq_processor,
                num_samples=args.text_samples,
                batch_size=args.text_batch_size,
                device=DEVICE,
            )
            
            print("\nRunning text benchmark for AWQ model...")
            awq_text_stats = run_benchmark(
                awq_model,
                text_batches,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                scenario_name="text_queries",
            )
            print_results("AWQ", awq_text_stats)
            all_results["AWQ_text_queries"] = awq_text_stats
            
            del text_batches
            gc.collect()
            torch.cuda.empty_cache()
        
        if not args.text_only and not batch_sizes:
            print("Preparing image batches for AWQ model...")
            image_batches = prepare_image_batches(
                awq_processor,
                num_samples=args.image_samples,
                batch_size=args.image_batch_size,
                image_size=args.image_size,
                device=DEVICE,
            )
            
            print("\nRunning image benchmark for AWQ model...")
            awq_image_stats = run_benchmark(
                awq_model,
                image_batches,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                scenario_name="image_documents",
            )
            print_results("AWQ", awq_image_stats)
            all_results["AWQ_image_documents"] = awq_image_stats
            
            del image_batches
            gc.collect()
            torch.cuda.empty_cache()
        
        cleanup_model(awq_model)
        del awq_processor
        gc.collect()
        torch.cuda.empty_cache()
    
    if not args.only_base and not args.only_awq:
        if batch_sizes and base_sweep_results and awq_sweep_results:
            print_sweep_summary(base_sweep_results, awq_sweep_results)
        else:
            print_summary(all_results)
    
    if args.output_json:
        meta = {
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "text_samples": args.text_samples,
            "text_batch_size": args.text_batch_size,
            "image_samples": args.image_samples,
            "image_batch_size": args.image_batch_size,
            "image_size": args.image_size,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
        }
        output = {"metadata": meta, "results": all_results}
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")
    
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
