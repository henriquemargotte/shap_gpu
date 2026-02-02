from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - informative import
    torch = None

# Lightweight standalone helper: avoid importing the full `shap` package
# so this module can be copied/run standalone in minimal environments.


class GPUTextMaskedModel:
    """A lightweight GPU helper to build masked token batches and run a HF/PyTorch model on CUDA.

    Limitations / assumptions:
    - The wrapped model is a PyTorch model that accepts `input_ids` and `attention_mask` tensors.
    - Inputs are fixed-length token id arrays (padded to same `seq_len`).
    - `background_input_ids` is an ``np.ndarray`` of shape `(B, seq_len)` used as background rows.
    - `x_input_ids` is a 1D array of length `seq_len` for the example being explained.
    - This helper averages model outputs across the `B` backgrounds for each mask.
    """

    def __init__(
        self,
        model: Any,
        device: str,
        background_input_ids: np.ndarray,
        pad_token_id: int | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for GPUTextMaskedModel")

        self.model = model
        self.device = torch.device(device if device is not None else "cuda")
        self.bg = torch.tensor(background_input_ids, dtype=torch.long, device=self.device)
        self.B, self.seq_len = self.bg.shape
        self.pad_token_id = pad_token_id

    def _build_masked_batch(self, x_input_ids: np.ndarray, masks: np.ndarray) -> torch.Tensor:
        # masks: (M, seq_len) boolean, True -> keep x token, False -> use background
        M = masks.shape[0]
        x = torch.tensor(x_input_ids, dtype=torch.long, device=self.device)
        masks_t = torch.tensor(masks, dtype=torch.bool, device=self.device)

        # expand x and background to (M, B, seq_len)
        x_expand = x.unsqueeze(0).unsqueeze(0).expand(M, self.B, self.seq_len)
        bg_expand = self.bg.unsqueeze(0).expand(M, self.B, self.seq_len)
        masks_expand = masks_t.unsqueeze(1).expand(M, self.B, self.seq_len)

        masked = torch.where(masks_expand, x_expand, bg_expand)
        return masked.reshape(M * self.B, self.seq_len)

    def _build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return (input_ids != self.pad_token_id).long()

    def __call__(self, x_input_ids: np.ndarray, masks: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Evaluate the model on all masks and average over backgrounds.

        Args:
            x_input_ids: (seq_len,) numpy array of token ids for the explained example.
            masks: (M, seq_len) boolean numpy array. True keeps token from x, False uses background token.
            batch_size: model forward batch size (number of rows per forward pass).

        Returns:
            averaged_outs: numpy array of shape (M, ...) where ... is model output dimension.
        """
        masked_flat = self._build_masked_batch(x_input_ids, masks)
        attn = self._build_attention_mask(masked_flat)

        outs_list = []
        with torch.no_grad():
            for i in range(0, masked_flat.shape[0], batch_size):
                batch_ids = masked_flat[i : i + batch_size]
                batch_attn = attn[i : i + batch_size]
                # some models expect dict input, adapt as needed
                model_inputs = {"input_ids": batch_ids, "attention_mask": batch_attn}
                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
                out = self.model(**model_inputs)
                # try to get logits/tensor output consistently
                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                else:
                    out = np.asarray(out)
                outs_list.append(out)

        outputs = np.concatenate(outs_list, axis=0)
        M = masks.shape[0]
        outputs = outputs.reshape(M, self.B, -1)
        return outputs.mean(axis=1)


class PermutationExplainerGPU:
    """A minimal GPU-based permutation explainer for token-level text models.

    Usage: replace `PermutationExplainer` with this class when your model is a
    PyTorch/HuggingFace model and you have tokenized background examples.
    This implementation focuses on vectorized mask application and batched GPU inference.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        background_input_ids: np.ndarray,
        device: str = "cuda",
        pad_token_id: int | None = None,
        seed: int | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for PermutationExplainerGPU")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bg_input_ids = background_input_ids
        self.pad_token_id = pad_token_id
        self.seed = seed

    def explain_row(self, x: str | np.ndarray, npermutations: int = 10, batch_size: int = 256) -> dict:
        # tokenization: accept pre-tokenized ids or raw string
        if isinstance(x, np.ndarray):
            x_ids = x
        else:
            toks = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
            x_ids = toks["input_ids"][0].cpu().numpy()

        seq_len = x_ids.shape[0]
        if self.seed is not None:
            np.random.seed(self.seed)

        inds = np.arange(seq_len)
        nperms = npermutations
        all_masks = []
        for _ in range(nperms):
            np.random.shuffle(inds)
            # initial all-false mask (use background)
            mask = np.zeros(seq_len, dtype=bool)
            all_masks.append(mask.copy())
            # forward: progressively turn on tokens from x
            for ind in inds:
                mask = mask.copy()
                mask[ind] = True
                all_masks.append(mask.copy())
            # backward: progressively turn off tokens
            for ind in inds:
                mask = mask.copy()
                mask[ind] = False
                all_masks.append(mask.copy())

        all_masks = np.stack(all_masks, axis=0)
        gpu_model = GPUTextMaskedModel(self.model, self.device, self.bg_input_ids, pad_token_id=self.pad_token_id)
        outputs = gpu_model(x_ids, all_masks, batch_size=batch_size)

        # outputs shape (n_masks_total, out_dim)
        # compute shap-like values: difference between consecutive masks per permutation
        row_values = np.zeros((seq_len, outputs.shape[1]))
        nmask_block = 2 * seq_len + 1
        for p in range(nperms):
            base = p * nmask_block
            # forward diffs
            for i_idx in range(seq_len):
                idx = inds[i_idx]
                row_values[idx] += outputs[base + i_idx + 1] - outputs[base + i_idx]
            # backward diffs
            for i_idx in range(seq_len):
                idx = inds[i_idx]
                row_values[idx] += outputs[base + seq_len + 1 + i_idx] - outputs[base + seq_len + 1 + i_idx + 1]

        row_values = row_values / (2 * nperms)
        expected_value = outputs[0]

        return {"values": row_values, "expected_values": expected_value, "mask_shapes": [(seq_len,)], "clustering": None}
