#!/usr/bin/env python3
"""Benchmark SHAP explainer CPU vs GPU for a tabular classification problem.

Creates a CPU model and (optionally) a GPU model using cuML when
available, runs SHAP explanations on the same input set for both, and
reports timing breakdown so the user can identify bottlenecks.

Usage:
  python scripts/run_shap_tabular_device_benchmark.py --n-samples 200
  python scripts/run_shap_tabular_device_benchmark.py --n-samples 200 --full

Notes:
- Uses `sklearn.datasets.load_breast_cancer` as a local tabular dataset.
- If `cuml` is installed the script will attempt a GPU run; otherwise
  GPU run is skipped.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import shap

try:
    import cupy as cp
    from cuml.explainer import PermutationExplainer as CuMPermutationExplainer
    has_cuml_explainer = True
except Exception:
    cp = None
    CuMPermutationExplainer = None
    has_cuml_explainer = False

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TimingWrapper:
    """Wraps a prediction function and tracks cumulative inference time."""

    def __init__(self, func):
        self.func = func
        self.total_time = 0.0
        self.call_count = 0

    def __call__(self, X, **kwargs):
        t0 = time.perf_counter()
        out = self.func(X, **kwargs)
        t1 = time.perf_counter()
        self.total_time += (t1 - t0)
        self.call_count += 1
        return out


def train_cpu_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model


def try_train_gpu_model(X_train, y_train):
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as cuRF
    except Exception:
        return None

    try:
        # cuML accepts numpy arrays; train on the GPU-backed implementation
        gmodel = cuRF(n_estimators=100, random_state=0)
        gmodel.fit(X_train, y_train)
        return gmodel
    except Exception:
        return None


def run_cuml_permutation(cuml_model_predict, X_train_cu, X_explain_cu, n_samples=None):
    # cuML PermutationExplainer expects GPU arrays; measure init and execution separately
    wrapper = TimingWrapper(cuml_model_predict)
    t0_init = time.perf_counter()
    explainer = CuMPermutationExplainer(model=wrapper, data=X_train_cu, random_state=42)
    t1_init = time.perf_counter()

    t0 = time.perf_counter()
    vals = explainer.shap_values(X_explain_cu if n_samples is None else X_explain_cu[:n_samples])
    t1 = time.perf_counter()

    return {
        "explainer_init_time": t1_init - t0_init,
        "explainer_total_time": t1 - t0,
        "model_cumulative_time": wrapper.total_time,
        "model_call_count": wrapper.call_count,
        "result": vals,
    }


def run_explainer_and_time(wrapper, X, background=None):
    # Create a simple Independent masker using the provided background and measure init times
    masker = None
    masker_time = 0.0
    if background is not None:
        t0_mask = time.perf_counter()
        try:
            masker = shap.maskers.Independent(background)
        except Exception:
            masker = None
        t1_mask = time.perf_counter()
        masker_time = t1_mask - t0_mask

    t0_init = time.perf_counter()
    if masker is not None:
        explainer = shap.Explainer(wrapper, masker)
    else:
        explainer = shap.Explainer(wrapper)
    t1_init = time.perf_counter()

    t0 = time.perf_counter()
    vals = explainer(X)
    t1 = time.perf_counter()

    timing = {
        "masker_init_time": masker_time,
        "explainer_init_time": t1_init - t0_init,
        "explainer_total_time": t1 - t0,
    }
    return vals, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200, help="number of dataset samples to explain (default 200)")
    parser.add_argument("--full", action="store_true", help="use the whole dataset (overrides --n-samples)")
    parser.add_argument("--out", type=str, default="shap_tabular_device_benchmark_results.json")
    parser.add_argument("--big", action="store_true", help="use a synthetic bigger dataset")
    parser.add_argument("--n-train", type=int, default=5000, help="when --big, number of training rows to generate (default 5000)")
    parser.add_argument("--n-features", type=int, default=100, help="when --big, number of features to generate (default 100)")
    parser.add_argument("--background-size", type=int, default=100, help="number of background rows passed to the masker (default 100)")
    args = parser.parse_args()

    if args.big:
        print(f"Generating synthetic dataset (train={args.n_train}, features={args.n_features})...")
        n_total = args.n_train + 2000
        X, y = make_classification(n_samples=n_total, n_features=args.n_features, n_informative=max(2, args.n_features // 10), random_state=0)
        dataset_name = f"synthetic_{n_total}x{args.n_features}"
    else:
        print("Loading breast cancer dataset...")
        data = load_breast_cancer()
        X = data.data
        y = data.target
        dataset_name = "breast_cancer"

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    if args.full:
        n = len(X_test)
    else:
        n = min(args.n_samples, len(X_test))
    print(f"Using {n} samples for explanation")
    X_subset = X_test[:n]

    # Train CPU model
    print("Training CPU model (sklearn RandomForest)...")
    cpu_model = train_cpu_model(X_train, y_train)

    def cpu_predict_proba(X_in, **kwargs):
        return cpu_model.predict_proba(np.asarray(X_in))

    cpu_wrapper = TimingWrapper(cpu_predict_proba)

    # warmup
    _ = cpu_wrapper(X_subset[:2])

    # Attempt GPU model via cuML
    gpu_model = try_train_gpu_model(X_train, y_train)
    gpu_available = gpu_model is not None
    if not gpu_available:
        print("cuML not available or GPU model train failed; GPU run will be skipped.")
    else:
        print("GPU model trained using cuML; preparing GPU wrapper...")

        def gpu_predict_proba(X_in, **kwargs):
            try:
                return gpu_model.predict_proba(np.asarray(X_in))
            except Exception:
                # some cuML versions return a different shape/type; coerce to numpy
                return np.asarray(gpu_model.predict_proba(np.asarray(X_in)))

        gpu_wrapper = TimingWrapper(gpu_predict_proba)
        _ = gpu_wrapper(X_subset[:2])

    results = {"n_samples": n, "dataset": dataset_name, "cpu": {}, "gpu": {}, "cuml_native": {}}

    # Baseline full-dataset prediction times
    print("Running baseline full-dataset prediction (CPU)...")
    t0 = time.perf_counter()
    _ = cpu_model.predict_proba(X_test)
    t1 = time.perf_counter()
    results["cpu"]["baseline_pred_time"] = t1 - t0

    if gpu_available:
        print("Running baseline full-dataset prediction (GPU)...")
        t0 = time.perf_counter()
        _ = gpu_model.predict_proba(X_test)
        t1 = time.perf_counter()
        results["gpu"]["baseline_pred_time"] = t1 - t0

    # Run SHAP explainer and time
    print("Running SHAP explainer (CPU model)...")
    cpu_wrapper.total_time = 0.0
    cpu_wrapper.call_count = 0
    bg_size = min(args.background_size, len(X_train))
    vals_cpu, cpu_timing = run_explainer_and_time(cpu_wrapper, X_subset, background=X_train[:bg_size])
    results["cpu"]["explainer_total_time"] = cpu_timing["explainer_total_time"]
    results["cpu"]["explainer_init_time"] = cpu_timing["explainer_init_time"]
    results["cpu"]["masker_init_time"] = cpu_timing["masker_init_time"]
    results["cpu"]["model_cumulative_time"] = cpu_wrapper.total_time
    results["cpu"]["model_call_count"] = cpu_wrapper.call_count

    if gpu_available:
        print("Running SHAP explainer (GPU model)...")
        gpu_wrapper.total_time = 0.0
        gpu_wrapper.call_count = 0
        bg_size = min(args.background_size, len(X_train))
        vals_gpu, gpu_timing = run_explainer_and_time(gpu_wrapper, X_subset, background=X_train[:bg_size])
        results["gpu"]["explainer_total_time"] = gpu_timing["explainer_total_time"]
        results["gpu"]["explainer_init_time"] = gpu_timing["explainer_init_time"]
        results["gpu"]["masker_init_time"] = gpu_timing["masker_init_time"]
        results["gpu"]["model_cumulative_time"] = gpu_wrapper.total_time
        results["gpu"]["model_call_count"] = gpu_wrapper.call_count

        # If cuML's native PermutationExplainer is available, run it as a third path
        if has_cuml_explainer and cp is not None:
            try:
                print("Running cuML native PermutationExplainer (GPU)...")
                # prepare GPU arrays (float32) for cuML explainer and measure copy time
                t0_copy = time.perf_counter()
                X_train_cu = cp.asarray(X_train.astype("float32"))
                X_subset_cu = cp.asarray(X_subset.astype("float32"))
                t1_copy = time.perf_counter()

                # wrap cuml model predict_proba if necessary
                def cuml_predict_proba(X_in_cu, **kwargs):
                    return gpu_model.predict_proba(X_in_cu)

                cu_res = run_cuml_permutation(cuml_predict_proba, X_train_cu, X_subset_cu, n_samples=None)
                results["cuml_native"].update({k: v for k, v in cu_res.items() if k != "result"})
                results["cuml_native"]["data_copy_time"] = t1_copy - t0_copy
            except Exception as e:
                results["cuml_native"]["error"] = str(e)
        else:
            results["cuml_native"]["available"] = False

    # compute overheads
    results["cpu"]["explainer_overhead_time"] = results["cpu"]["explainer_total_time"] - results["cpu"]["model_cumulative_time"]
    if gpu_available:
        results["gpu"]["explainer_overhead_time"] = results["gpu"]["explainer_total_time"] - results["gpu"]["model_cumulative_time"]
    if "explainer_total_time" in results["cuml_native"]:
        results["cuml_native"]["explainer_overhead_time"] = results["cuml_native"]["explainer_total_time"] - results["cuml_native"]["model_cumulative_time"]

    # report
    def print_summary():
        print("\nBenchmark summary:")
        print(f"Samples: {n}")
        print("\nCPU run:")
        print(f"  explainer total: {results['cpu']['explainer_total_time']:.3f}s")
        print(f"  explainer init: {results['cpu'].get('explainer_init_time', 0.0):.3f}s")
        print(f"  masker init: {results['cpu'].get('masker_init_time', 0.0):.3f}s")
        print(f"  model cumulative (measured): {results['cpu']['model_cumulative_time']:.3f}s")
        print(f"  explainer overhead (SHAP pre/post): {results['cpu']['explainer_overhead_time']:.3f}s")
        if gpu_available:
            print("\nGPU run:")
            print(f"  explainer total: {results['gpu']['explainer_total_time']:.3f}s")
            print(f"  explainer init: {results['gpu'].get('explainer_init_time', 0.0):.3f}s")
            print(f"  masker init: {results['gpu'].get('masker_init_time', 0.0):.3f}s")
            print(f"  model cumulative (measured): {results['gpu']['model_cumulative_time']:.3f}s")
            print(f"  explainer overhead (SHAP pre/post): {results['gpu']['explainer_overhead_time']:.3f}s")
            speedup = results['cpu']['model_cumulative_time'] / max(1e-9, results['gpu']['model_cumulative_time'])
            print(f"\nModel cumulative speedup (CPU_time / GPU_time): {speedup:.2f}x")
            total_ratio = results['cpu']['explainer_total_time'] / max(1e-9, results['gpu']['explainer_total_time'])
            print(f"Explainer total time ratio (CPU / GPU): {total_ratio:.2f}x")
            if "explainer_total_time" in results["cuml_native"]:
                print("\ncuML native PermutationExplainer run:")
                print(f"  data copy time: {results['cuml_native'].get('data_copy_time', 0.0):.3f}s")
                print(f"  explainer init: {results['cuml_native'].get('explainer_init_time', 0.0):.3f}s")
                print(f"  explainer total: {results['cuml_native']['explainer_total_time']:.3f}s")
                print(f"  model cumulative (measured): {results['cuml_native']['model_cumulative_time']:.3f}s")
                print(f"  explainer overhead (SHAP pre/post): {results['cuml_native']['explainer_overhead_time']:.3f}s")
                speedup_cuml = results['cpu']['model_cumulative_time'] / max(1e-9, results['cuml_native']['model_cumulative_time'])
                print(f"\nModel cumulative speedup (CPU_time / cuML_native_time): {speedup_cuml:.2f}x")
                total_ratio_cuml = results['cpu']['explainer_total_time'] / max(1e-9, results['cuml_native']['explainer_total_time'])
                print(f"Explainer total time ratio (CPU / cuML_native): {total_ratio_cuml:.2f}x")
        else:
            print("GPU not available; only CPU results shown.")

    print_summary()

    print("\nFull results (JSON):")
    print(json.dumps(results, indent=2))

    out_path = Path(args.out)
    try:
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Results written to {out_path}")
    except Exception:
        print(f"Could not write results to {out_path}; printed to stdout instead.")


if __name__ == "__main__":
    main()
