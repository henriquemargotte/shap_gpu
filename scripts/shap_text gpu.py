from datasets import load_dataset
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
import tempfile
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import shap

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
    
def _build_cudf_tokenizer(hf_tokenizer):
    try:
        from cudf.core.subword_tokenizer import SubwordTokenizer, TokenizerVocabulary
    except Exception:
        return None

    if hasattr(TokenizerVocabulary, "from_pretrained"):
        vocab = TokenizerVocabulary.from_pretrained(hf_tokenizer.name_or_path)
        return SubwordTokenizer(vocab)

    vocab_items = hf_tokenizer.get_vocab() if hasattr(hf_tokenizer, "get_vocab") else hf_tokenizer.vocab
    max_id = max(vocab_items.values())
    inv_vocab = [""] * (max_id + 1)
    for token, idx in vocab_items.items():
        inv_vocab[idx] = token

    vocab_path = Path(tempfile.mkdtemp()) / "vocab.txt"
    vocab_path.write_text("\n".join(inv_vocab))
    do_lower_case = bool(getattr(hf_tokenizer, "do_lower_case", False))
    vocab = TokenizerVocabulary(str(vocab_path), do_lower_case=do_lower_case)
    return SubwordTokenizer(vocab)


def _call_cudf_tokenizer(cudf_tokenizer, series, max_length):
    tokenize_fn = cudf_tokenizer.tokenize if hasattr(cudf_tokenizer, "tokenize") else cudf_tokenizer
    call_kwargs = [
        {"max_length": max_length, "truncation": True, "add_special_tokens": True},
        {"max_length": max_length, "truncation": True},
        {"max_length": max_length},
        {},
    ]
    for kwargs in call_kwargs:
        try:
            return tokenize_fn(series, **kwargs)
        except TypeError:
            continue
    return tokenize_fn(series)


def _cudf_tokenize_batch(cudf_tokenizer, texts, max_length, pad_token_id):
    import cudf

    series = cudf.Series(texts)
    tokenized = _call_cudf_tokenizer(cudf_tokenizer, series, max_length)
    if isinstance(tokenized, dict):
        token_ids = tokenized.get("input_ids", tokenized)
    else:
        token_ids = getattr(tokenized, "input_ids", tokenized)

    if hasattr(token_ids, "to_pandas"):
        token_lists = token_ids.to_pandas().tolist()
    elif hasattr(token_ids, "to_arrow"):
        token_lists = token_ids.to_arrow().to_pylist()
    elif hasattr(token_ids, "tolist"):
        token_lists = token_ids.tolist()
    else:
        token_lists = list(token_ids)

    pad_value = 0 if pad_token_id is None else pad_token_id
    padded = []
    for ids in token_lists:
        ids = [] if ids is None else list(ids)
        if len(ids) > max_length:
            ids = ids[:max_length]
        if len(ids) < max_length:
            ids = ids + [pad_value] * (max_length - len(ids))
        padded.append(ids)
    return padded


def setup_model_and_data():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
    cudf_tokenizer = _build_cudf_tokenizer(tokenizer)

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

    return model, tokenizer, cudf_tokenizer, data, labels

def f(x, model, tokenizer, cudf_tokenizer=None, max_length=128):
    if cudf_tokenizer is not None:
        input_ids = _cudf_tokenize_batch(cudf_tokenizer, x, max_length, tokenizer.pad_token_id)
    else:
        input_ids = [
            tokenizer.encode(v, padding="max_length", max_length=max_length, truncation=True)
            for v in x
        ]
    tv = torch.tensor(input_ids)
    attention_mask = (tv != 0).type(torch.int64)
    outputs = model(tv, attention_mask=attention_mask)[0].detach().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    return val

def main():
    model, tokenizer, cudf_tokenizer, data, labels = setup_model_and_data()
    if cudf_tokenizer is None:
        print("cuDF tokenizer not available; using HF tokenizer for encoding")

    # Wrap model prediction function with timing
    wrapped_f = TimingWrapper(lambda x: f(x, model, tokenizer, cudf_tokenizer=cudf_tokenizer))

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