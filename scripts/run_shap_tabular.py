"""Tabular permutation SHAP benchmark.

Compares:
 - SHAP Python `PermutationExplainer` (or `shap.Explainer(..., algorithm="permutation")`) using a CPU model
 - cuML `PermutationExplainer` (GPU) when available

The script trains a small regression problem, times baseline prediction, explainer total time,
and cumulative model prediction time (measured by a thin wrapper) for each path.
"""
import argparse
import json
import time
import numpy as np
import warnings

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR as SklearnSVR

import shap

try:
    from cuml import SVR as CuMLSvr
    from cuml.explainer import PermutationExplainer as CuMPermutationExplainer
    import cupy as cp
    has_cuml = True
except Exception:
    CuMLSvr = None
    CuMPermutationExplainer = None
    cp = None
    has_cuml = False


class TimingWrapper:
    """Wraps a callable and accumulates inference time and calls."""

    def __init__(self, func):
        self.func = func
        self.total_time = 0.0
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self.func(*args, **kwargs)
        t1 = time.perf_counter()
        self.total_time += (t1 - t0)
        self.call_count += 1
        return out


def run_shap_python_permutation(model_predict, X_train, X_explain, n_samples=None):
    masker = shap.maskers.Independent(X_train)
    wrapper = TimingWrapper(model_predict)
    explainer = shap.Explainer(wrapper, masker, algorithm="permutation")
    t0 = time.perf_counter()
    res = explainer(X_explain if n_samples is None else X_explain[:n_samples])
    t1 = time.perf_counter()
    return {
        "explainer_total_time": t1 - t0,
        "model_cumulative_time": wrapper.total_time,
        "model_call_count": wrapper.call_count,
        "result": res,
    }


def run_cuml_permutation(cuml_model_predict, X_train_cu, X_explain_cu, n_samples=None):
    # cuML PermutationExplainer expects GPU arrays; timing is measured around shap_values call
    wrapper = TimingWrapper(cuml_model_predict)
    explainer = CuMPermutationExplainer(model=wrapper, data=X_train_cu, random_state=42)
    t0 = time.perf_counter()
    # cuml explainer returns array-like
    vals = explainer.shap_values(X_explain_cu if n_samples is None else X_explain_cu[:n_samples])
    t1 = time.perf_counter()
    return {
        "explainer_total_time": t1 - t0,
        "model_cumulative_time": wrapper.total_time,
        "model_call_count": wrapper.call_count,
        "result": vals,
    }


def run_shap_python_on_gpu_model(cuml_model, X_train, X_explain, n_samples=None):
    """Run SHAP Python permutation explainer but using a GPU-backed cuML model.

    The SHAP masker and explainer operate on numpy arrays; this helper converts
    numpy -> cupy before calling the cuML model and converts outputs back.
    Timing measures the full model.predict call including conversion.
    """
    if cp is None:
        raise RuntimeError("cupy is not available for GPU model wrapping")

    def model_predict_numpy_to_gpu(x_numpy):
        # ensure float32 for cuML
        x_cu = cp.asarray(x_numpy.astype('float32'))
        y_cu = cuml_model.predict(x_cu)
        # convert back to numpy for SHAP
        return cp.asnumpy(y_cu)

    wrapper = TimingWrapper(model_predict_numpy_to_gpu)
    masker = shap.maskers.Independent(X_train)
    explainer = shap.Explainer(wrapper, masker, algorithm="permutation")
    t0 = time.perf_counter()
    res = explainer(X_explain if n_samples is None else X_explain[:n_samples])
    t1 = time.perf_counter()
    return {
        "explainer_total_time": t1 - t0,
        "model_cumulative_time": wrapper.total_time,
        "model_call_count": wrapper.call_count,
        "result": res,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50, help="number of samples to explain (default 50)")
    parser.add_argument("--n-features", type=int, default=10, help="number of features (default 10)")
    parser.add_argument("--n-train", type=int, default=2000, help="training samples (default 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cuml", action="store_true", help="skip cuML GPU benchmark even if available")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # generate a moderately sized regression problem
    X, y = make_regression(n_samples=args.n_train + args.n_samples, n_features=args.n_features, noise=0.1, random_state=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.n_samples, random_state=args.seed)

    results = {"n_features": args.n_features, "n_train": X_train.shape[0], "n_explain": X_test.shape[0], "python_shap": {}, "cuml": {}}

    # Train sklearn CPU model
    cpu_model = SklearnSVR()
    t0 = time.perf_counter()
    cpu_model.fit(X_train, y_train)
    t1 = time.perf_counter()
    results["python_shap"]["cpu_model_train_time"] = t1 - t0

    # Baseline prediction time (CPU)
    t0 = time.perf_counter()
    _ = cpu_model.predict(X_test)
    t1 = time.perf_counter()
    results["python_shap"]["baseline_pred_time_cpu"] = t1 - t0

    # Run SHAP Python permutation explainer (this runs the SHAP sampling/aggregation in Python)
    print("Running SHAP Python PermutationExplainer (CPU model)...")
    py_res = run_shap_python_permutation(cpu_model.predict, X_train, X_test, n_samples=None)
    results["python_shap"].update({k: v for k, v in py_res.items() if k != "result"})

    # Optionally run cuML benchmark
    if has_cuml and not args.skip_cuml:
        try:
            print("cuML detected — preparing GPU model and data...")
            # convert numpy arrays to cupy
            X_train_cu = cp.asarray(X_train)
            X_test_cu = cp.asarray(X_test)
            y_train_cu = cp.asarray(y_train)

            cuml_model = CuMLSvr()
            t0 = time.perf_counter()
            cuml_model.fit(X_train_cu, y_train_cu)
            t1 = time.perf_counter()
            results["cuml"]["gpu_model_train_time"] = t1 - t0

            # baseline prediction time on GPU
            t0 = time.perf_counter()
            _ = cuml_model.predict(X_test_cu)
            t1 = time.perf_counter()
            results["cuml"]["baseline_pred_time_gpu"] = t1 - t0

            print("Running cuML PermutationExplainer (GPU)...")
            cu_res = run_cuml_permutation(cuml_model.predict, X_train_cu, X_test_cu, n_samples=None)
            results["cuml"].update({k: v for k, v in cu_res.items() if k != "result"})
            # Also run SHAP Python permutation explainer using the cuML GPU model
            try:
                print("Running SHAP Python PermutationExplainer but using cuML GPU model (numpy->cupy conversions)...")
                py_gpu_res = run_shap_python_on_gpu_model(cuml_model, X_train, X_test, n_samples=None)
                results["python_shap"]["gpu_model"] = {k: v for k, v in py_gpu_res.items() if k != "result"}
            except Exception as e:
                warnings.warn(f"SHAP-on-GPU run failed: {e}")
                results["python_shap"]["gpu_model_error"] = str(e)
        except Exception as e:
            warnings.warn(f"cuML benchmark failed: {e}")
            results["cuml"]["error"] = str(e)
    else:
        results["cuml"]["available"] = False

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()