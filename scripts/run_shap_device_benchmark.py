#!/usr/bin/env python3
"""Benchmark SHAP explainer CPU vs GPU for a transformers sentiment pipeline.

Creates two identical pipelines (same checkpoint) on CPU and GPU, runs
SHAP explanations on the same input set for both, and reports timing
breakdown so the user can identify bottlenecks.

Usage:
  python scripts/run_shap_device_benchmark.py --n-samples 200
  python scripts/run_shap_device_benchmark.py --n-samples 200 --full

Notes:
- By default the script uses a subset (`--n-samples`) to keep runtimes
  reasonable; pass `--n-samples` equal to the dataset size to run on
  the whole dataset. The core timing comparison isolates the SHAP
  explainer invocation (total explainer time) and the cumulative model
  inference time measured by a thin wrapper around the pipeline.
"""
import argparse
import json
import time
from pathlib import Path

import datasets
import transformers
import shap


class TimingWrapper:
    """Wraps a transformers pipeline and tracks cumulative inference time."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.total_time = 0.0
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        """Accept arbitrary args (masker may call with multiple args).

        Returns whatever the wrapped model returns and accumulates time.
        """
        t0 = time.perf_counter()
        out = self.pipeline(*args, **kwargs)
        t1 = time.perf_counter()
        self.total_time += (t1 - t0)
        self.call_count += 1
        return out


def make_pipeline(device, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    # device=-1 -> CPU, device=0 -> first CUDA device
    return transformers.pipeline(
        "sentiment-analysis", model=model_name, tokenizer=model_name, return_all_scores=True, device=device
    )


def run_explainer_and_time(wrapper, data, tokenizer=None):
    # Create a Text masker from the provided tokenizer when available.
    masker = None
    if tokenizer is not None:
        try:
            masker = shap.maskers.Text(tokenizer)
        except Exception:
            masker = None

    if masker is not None:
        explainer = shap.Explainer(wrapper, masker)
    else:
        explainer = shap.Explainer(wrapper)

    t0 = time.perf_counter()
    vals = explainer(data)
    t1 = time.perf_counter()
    total_explainer_time = t1 - t0
    return vals, total_explainer_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200, help="number of dataset samples to explain (default 200)")
    parser.add_argument("--full", action="store_true", help="use the whole dataset (overrides --n-samples)")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device id to use for GPU pipeline")
    parser.add_argument("--out", type=str, default="shap_device_benchmark_results.json")
    args = parser.parse_args()

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    print("Loading IMDB test dataset...")
    dataset = datasets.load_dataset("imdb", split="test")
    if args.full:
        n = len(dataset)
    else:
        n = min(args.n_samples, len(dataset))
    print(f"Using {n} samples for explanation")
    texts = [t[:500] for t in dataset["text"][:n]]

    # create identical pipelines on CPU and GPU (if available)
    print("Creating CPU pipeline...")
    cpu_pipe = make_pipeline(device=-1, model_name=model_name)
    # create a SHAP-aware TransformersPipeline wrapper for masked calls
    cpu_pmodel = shap.models.TransformersPipeline(cpu_pipe, rescale_to_logits=True)
    cpu_wrapper = TimingWrapper(cpu_pmodel)

    # quick same-model warmup on CPU
    _ = cpu_wrapper(texts[:2])

    import torch

    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        print("CUDA not available on this machine; GPU run will be skipped.")
    else:
        print("Creating GPU pipeline (this will load the same checkpoint onto GPU)...")
        gpu_pipe = make_pipeline(device=args.gpu_id, model_name=model_name)
        gpu_pmodel = shap.models.TransformersPipeline(gpu_pipe, rescale_to_logits=True)
        gpu_wrapper = TimingWrapper(gpu_pmodel)
        # warmup
        # warmup using raw pipeline to avoid masked-call mismatch
        _ = gpu_pipe(texts[:2])

    results = {"n_samples": n, "model": model_name, "cpu": {}, "gpu": {}}

    # Baseline check: run a single full-dataset prediction pass on CPU and (if available) GPU
    print("Running baseline full-dataset prediction (CPU)...")
    t0 = time.perf_counter()
    _ = cpu_pipe(texts)
    t1 = time.perf_counter()
    baseline_cpu_pred_time = t1 - t0
    results["cpu"]["baseline_pred_time"] = baseline_cpu_pred_time

    if gpu_available:
        print("Running baseline full-dataset prediction (GPU)...")
        t0 = time.perf_counter()
        _ = gpu_pipe(texts)
        t1 = time.perf_counter()
        baseline_gpu_pred_time = t1 - t0
        results["gpu"]["baseline_pred_time"] = baseline_gpu_pred_time

    # Now run SHAP explainer and time the total explainer runtime and model-inference runtime
    print("Running SHAP explainer (CPU pipeline)...")
    # reset counters to measure only explainer phase
    cpu_wrapper.total_time = 0.0
    cpu_wrapper.call_count = 0
    vals_cpu, total_cpu = run_explainer_and_time(cpu_wrapper, texts, tokenizer=cpu_pipe.tokenizer)
    results["cpu"]["explainer_total_time"] = total_cpu
    results["cpu"]["model_cumulative_time"] = cpu_wrapper.total_time
    results["cpu"]["model_call_count"] = cpu_wrapper.call_count

    if gpu_available:
        print("Running SHAP explainer (GPU pipeline)...")
        gpu_wrapper.total_time = 0.0
        gpu_wrapper.call_count = 0
        vals_gpu, total_gpu = run_explainer_and_time(gpu_wrapper, texts, tokenizer=gpu_pipe.tokenizer)
        results["gpu"]["explainer_total_time"] = total_gpu
        results["gpu"]["model_cumulative_time"] = gpu_wrapper.total_time
        results["gpu"]["model_call_count"] = gpu_wrapper.call_count

    # compute overheads
    results["cpu"]["explainer_overhead_time"] = results["cpu"]["explainer_total_time"] - results["cpu"]["model_cumulative_time"]
    if gpu_available:
        results["gpu"]["explainer_overhead_time"] = results["gpu"]["explainer_total_time"] - results["gpu"]["model_cumulative_time"]

    # report
    def print_summary():
        print("\nBenchmark summary:")
        print(f"Samples: {n}")
        print(f"Model: {model_name}")
        print("\nCPU run:")
        print(f"  explainer total: {results['cpu']['explainer_total_time']:.3f}s")
        print(f"  model cumulative (measured): {results['cpu']['model_cumulative_time']:.3f}s")
        print(f"  explainer overhead (SHAP pre/post): {results['cpu']['explainer_overhead_time']:.3f}s")
        if gpu_available:
            print("\nGPU run:")
            print(f"  explainer total: {results['gpu']['explainer_total_time']:.3f}s")
            print(f"  model cumulative (measured): {results['gpu']['model_cumulative_time']:.3f}s")
            print(f"  explainer overhead (SHAP pre/post): {results['gpu']['explainer_overhead_time']:.3f}s")
            speedup = results['cpu']['model_cumulative_time'] / max(1e-9, results['gpu']['model_cumulative_time'])
            print(f"\nModel cumulative speedup (CPU_time / GPU_time): {speedup:.2f}x")
            total_ratio = results['cpu']['explainer_total_time'] / max(1e-9, results['gpu']['explainer_total_time'])
            print(f"Explainer total time ratio (CPU / GPU): {total_ratio:.2f}x")
        else:
            print("GPU not available; only CPU results shown.")

    print_summary()

    # print results to stdout instead of writing to a file (avoids permission issues)
    print("\nFull results (JSON):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
