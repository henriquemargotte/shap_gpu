"""Example: run the GPU token-level permutation explainer with a Hugging Face model.

Run with:

    python scripts/example_permutation_gpu_text.py

This downloads a small model/tokenizer, builds a small background, and runs
`PermutationExplainerGPU.explain_row` for one example.
"""
from __future__ import annotations

import os
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception as e:  # pragma: no cover - runtime import
    raise RuntimeError("Please install torch and transformers to run this example") from e

from _permutation_gpu_text import PermutationExplainerGPU


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = os.environ.get("HF_EXAMPLE_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    model.to(device)

    # small background set of sentences (pre-tokenize and pad to same length)
    background_texts = [
        "This is a neutral example.",
        "I love this movie.",
        "This is terrible and I hated it.",
    ]
    tok_bg = tokenizer(background_texts, padding=True, truncation=True, return_tensors="pt")
    bg_input_ids = tok_bg["input_ids"].numpy()
    max_len = bg_input_ids.shape[1]

    expl = PermationExplainerGPU = None
    expl = PermutationExplainerGPU(
        model=model,
        tokenizer=tokenizer,
        background_input_ids=bg_input_ids,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    # example to explain
    text = "The movie had beautiful visuals but the plot was boring."

    # pre-tokenize the text to the same length as the background to avoid shape mismatches
    toks_x = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    x_ids = toks_x["input_ids"][0].cpu().numpy()

    res = expl.explain_row(x_ids, npermutations=2, batch_size=64)

    print("values shape:", res["values"].shape)
    print("expected_values shape:", res["expected_values"].shape)
    print("First token contributions (first 5 tokens):\n", res["values"][0:5])


if __name__ == "__main__":
    main()
