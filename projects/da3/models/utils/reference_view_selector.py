# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reference view selection strategies for multi-view processing."""

import logging
import torch

logger = logging.getLogger("da3")


def cos_similarity(x, y):
    """Compute cosine similarity between two embeddings."""
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


def select_reference_view(
    x: torch.Tensor,
    strategy: str = "saddle_balanced",
) -> torch.Tensor:
    """
    Select reference view index per batch.

    Args:
        x: tensor of shape (B, S, N, C), typically the class tokens or features
        strategy: selection strategy

    Returns:
        b_idx: tensor of shape (B,) with the selected reference view index per batch
    """
    B, S, N, C = x.shape
    cls_features = x[:, :, 0, :]  # (B, S, C)

    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    elif strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)
    elif strategy in ("saddle_balanced", "saddle_sim_range"):
        return _saddle_selection(cls_features, strategy)
    else:
        raise ValueError(f"Unknown reference view strategy: {strategy}")


def _saddle_selection(cls_features: torch.Tensor, strategy: str) -> torch.Tensor:
    """Select reference view using saddle-point analysis of feature similarity."""
    B, S, C = cls_features.shape
    device = cls_features.device

    # Compute pairwise cosine similarity matrix
    features_norm = torch.nn.functional.normalize(cls_features, dim=-1)
    sim_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))  # (B, S, S)

    # For each view, compute the mean similarity to all other views
    mask = ~torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
    mean_sim = (sim_matrix * mask.float()).sum(dim=-1) / (S - 1)  # (B, S)

    if strategy == "saddle_balanced":
        # Select the view with the most balanced similarity (closest to median)
        median_sim = mean_sim.median(dim=-1, keepdim=True).values
        balance_score = -(mean_sim - median_sim).abs()  # Higher is better
        b_idx = balance_score.argmax(dim=-1)
    elif strategy == "saddle_sim_range":
        # Select the view that maximizes the range of similarities
        max_sim = (sim_matrix * mask.float() + (~mask).float() * (-1)).max(dim=-1).values
        min_sim = (sim_matrix * mask.float() + (~mask).float() * 2).min(dim=-1).values
        sim_range = max_sim - min_sim
        b_idx = sim_range.argmax(dim=-1)
    else:
        raise ValueError(f"Unknown saddle strategy: {strategy}")

    return b_idx


def reorder_by_reference(
    x: torch.Tensor,
    ref_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Reorder the view dimension so that the reference view is first.

    Args:
        x: (B, S, ...) tensor
        ref_idx: (B,) reference view indices

    Returns:
        Reordered tensor with ref view at position 0
    """
    B, S = x.shape[:2]
    device = x.device

    # Create reordering index: ref first, then others in order
    all_indices = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # (B, S)
    # Remove ref from the sequence and prepend
    batch_indices = []
    for b in range(B):
        ref = ref_idx[b].item()
        others = [i for i in range(S) if i != ref]
        batch_indices.append(torch.tensor([ref] + others, device=device))
    reorder_idx = torch.stack(batch_indices, dim=0)  # (B, S)

    # Gather along dim=1
    expand_shape = [B, S] + [1] * (x.ndim - 2)
    expand_full = list(x.shape)
    expand_full[0] = B
    expand_full[1] = S
    idx = reorder_idx.view(*expand_shape).expand(*expand_full)
    return torch.gather(x, 1, idx)


def restore_original_order(
    x: torch.Tensor,
    ref_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of reorder_by_reference: restore original view order.

    Args:
        x: (B, S, ...) tensor (reordered)
        ref_idx: (B,) reference view indices used in reordering

    Returns:
        Tensor with original view order restored
    """
    B, S = x.shape[:2]
    device = x.device

    # Compute inverse permutation
    batch_indices = []
    for b in range(B):
        ref = ref_idx[b].item()
        forward = [ref] + [i for i in range(S) if i != ref]
        inv = [0] * S
        for new_pos, old_pos in enumerate(forward):
            inv[old_pos] = new_pos
        batch_indices.append(torch.tensor(inv, device=device))
    restore_idx = torch.stack(batch_indices, dim=0)  # (B, S)

    expand_shape = [B, S] + [1] * (x.ndim - 2)
    expand_full = list(x.shape)
    expand_full[0] = B
    expand_full[1] = S
    idx = restore_idx.view(*expand_shape).expand(*expand_full)
    return torch.gather(x, 1, idx)
