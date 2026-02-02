from cuml.datasets import make_blobs
import numpy as np
from cuml.model_selection import train_test_split
from cuml import KMeans as CuMLKMeans
from cuml.explainer import PermutationExplainer as CuMLPermutationExplainer
import shap
import cupy as cp
from sklearn.cluster import KMeans as SklearnKMeans

import time

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

def run_cuml_permutation(cuml_model_predict, X_train_cu, X_explain_cu, n_samples=None):
    # cuML PermutationExplainer expects GPU arrays; timing is measured around shap_values call
    wrapper = TimingWrapper(cuml_model_predict)
    explainer = CuMLPermutationExplainer(model=wrapper, data=X_train_cu, random_state=42)
    t0 = time.perf_counter()
    res = explainer.shap_values(X_explain_cu if n_samples is None else X_explain_cu[:n_samples])
    t1 = time.perf_counter()
    return {
        "explainer_total_time": t1 - t0,
        "model_cumulative_time": wrapper.total_time,
        "model_call_count": wrapper.call_count,
        "result": res,
    }

def run_permutation_explainer(model_predict, X_train, X_explain, n_samples=None):
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


def main():
    # Generate synthetic data
    X, y = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=42)
    X_train, X_explain, y_train, y_explain = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train cuML KMeans model
    cuml_model = CuMLKMeans(n_clusters=5, random_state=42)
    cuml_model.fit(X_train)

    # Convert data to GPU arrays
    X_train_cu = cp.asarray(X_train)
    X_explain_cu = cp.asarray(X_explain)

    # Train scikit-learn KMeans model
    sklearn_model = SklearnKMeans(n_clusters=5, random_state=42)
    sklearn_model.fit(X_train)

    # Run SHAP permutation explainer with scikit-learn model
    shap_results = run_permutation_explainer(sklearn_model.predict, X_train, X_explain, n_samples=100)

    # Run cuML permutation explainer
    cuml_results = run_cuml_permutation(cuml_model.predict, X_train_cu, X_explain_cu, n_samples=100)

    # Print results
    print("SHAP Python Permutation Explainer Results:")
    print(f"Explainer Total Time: {shap_results['explainer_total_time']:.4f} seconds")
    print(f"Model Cumulative Time: {shap_results['model_cumulative_time']:.4f} seconds")
    print(f"Model Call Count: {shap_results['model_call_count']}")

    print("cuML Permutation Explainer Results:")
    print(f"Explainer Total Time: {cuml_results['explainer_total_time']:.4f} seconds")
    print(f"Model Cumulative Time: {cuml_results['model_cumulative_time']:.4f} seconds")
    print(f"Model Call Count: {cuml_results['model_call_count']}")

if __name__ == "__main__":
    main()

# Samples 1000 points with 10 features and 5 centers.
# PermutationExplainer explainer: 101it [00:10,  2.49it/s]                        
# SHAP Python Permutation Explainer Results:
# Explainer Total Time: 10.8665 seconds
# Model Cumulative Time: 5.6350 seconds
# Model Call Count: 2399
# cuML Permutation Explainer Results:
# Explainer Total Time: 3.3750 seconds
# Model Cumulative Time: 0.9288 seconds
# Model Call Count: 1001

# Samples 5000 points with 30 features and 10 centers.
# SHAP Python Permutation Explainer Results:
# Explainer Total Time: 9.4879 seconds
# Model Cumulative Time: 4.3377 seconds
# Model Call Count: 899
# cuML Permutation Explainer Results:
# Explainer Total Time: 11.4270 seconds
# Model Cumulative Time: 6.5770 seconds
# Model Call Count: 1001