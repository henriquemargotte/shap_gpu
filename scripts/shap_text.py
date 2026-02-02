from datasets import load_dataset
import numpy as np
import pandas as pd
import scipy as sp
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import shap
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
    
def setup_model_and_data():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

    # Load dataset
    dataset = load_dataset("emotion")

    train = dataset["train"]
    test = dataset["test"]

    data = {"text": train["text"], "emotion": train["label"]}

    data = pd.DataFrame(data)

    # set mapping between label and id
    id2label = model.config.id2label
    label2id = model.config.label2id
    labels = sorted(label2id, key=label2id.get)

    return model, tokenizer, data, labels

def f(x, model, tokenizer):
    tv = torch.tensor([tokenizer.encode(v, padding="max_length", max_length=128, truncation=True) for v in x])
    attention_mask = (tv != 0).type(torch.int64)
    outputs = model(tv, attention_mask=attention_mask)[0].detach().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    return val

def main():
    model, tokenizer, data, labels = setup_model_and_data()

    # Wrap model prediction function with timing
    wrapped_f = TimingWrapper(lambda x: f(x, model, tokenizer))

    # Create SHAP explainer
    masker = shap.maskers.Text(tokenizer)
    explainer_permutation = shap.Explainer(wrapped_f, masker, output_names=labels, algorithm="permutation")
    explainer_default = shap.Explainer(wrapped_f, masker, output_names=labels)

    # Select subset of data to explain
    X_explain = data["text"].values[:20]

    # Compute SHAP values
    t0 = time.perf_counter()
    shap_values_permutation = explainer_permutation(X_explain)
    t1 = time.perf_counter()

    print(f"Permutation Explainer total time: {t1 - t0:.4f} seconds")
    print(f"Model cumulative time: {wrapped_f.total_time:.4f} seconds over {wrapped_f.call_count} calls")
    print(f"Time just for explainer: {t1 - t0 - wrapped_f.total_time:.4f} seconds")

    # Reset timing for default explainer
    wrapped_f.total_time = 0.0
    wrapped_f.call_count = 0
    t0 = time.perf_counter()
    shap_values_default = explainer_default(X_explain)
    t1 = time.perf_counter()

    #print(f"Explainer used: {explainer_default.algorithm}")
    print(f"Default Explainer total time: {t1 - t0:.4f} seconds")
    print(f"Model cumulative time: {wrapped_f.total_time:.4f} seconds over {wrapped_f.call_count} calls")
    print(f"Time just for explainer: {t1 - t0 - wrapped_f.total_time:.4f} seconds")

    #compare the shap values
    diff = np.abs(shap_values_permutation.values - shap_values_default.values).mean()
    print(f"Mean absolute difference between SHAP values: {diff:.6f}")

if __name__ == "__main__":
    main()

# PermutationExplainer explainer: 21it [09:03, 27.19s/it]                                                                                        
# Permutation Explainer total time: 543.7516 seconds
# Model cumulative time: 541.8794 seconds over 2196 calls
# Time just for explainer: 1.8722 seconds
# PartitionExplainer explainer: 21it [04:30, 14.23s/it]                                                                                          
# Default Explainer total time: 270.3535 seconds                                                                                                 
# Model cumulative time: 268.4267 seconds over 878 calls
# Time just for explainer: 1.9268 seconds