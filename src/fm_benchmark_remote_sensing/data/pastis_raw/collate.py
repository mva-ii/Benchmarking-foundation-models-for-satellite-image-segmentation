"""Collate functions for temporal sequence batching."""

from typing import Dict, List

import torch


def collate_temporal_sequences(batch: List[Dict]) -> Dict:
    """
    Collate batch of temporal sequences with padding for variable lengths.

    Input batch items are dicts with:
        - "data": (T, C, H, W) time series
        - "dates": (T,) day offsets
        - "masks": (H, W) segmentation labels
        - "pid": scalar patch ID

    Output batch dict:
        - "data": (B, T_max, C, H, W) padded time series
        - "dates": (B, T_max) padded dates
        - "masks": (B, H, W) segmentation labels
        - "pid": (B,) patch IDs
        - "pad_mask": (B, T_max) bool mask where True = padded
    """
    # Get max time length in batch
    max_time = max(item["data"].shape[0] for item in batch)

    batch_size = len(batch)
    c = batch[0]["data"].shape[1]
    h = batch[0]["data"].shape[2]
    w = batch[0]["data"].shape[3]

    # Initialize output tensors
    padded_data = torch.zeros(batch_size, max_time, c, h, w, dtype=torch.float32)
    padded_dates = torch.zeros(batch_size, max_time, dtype=torch.float32)
    pad_mask = torch.ones(batch_size, max_time, dtype=torch.bool)

    masks_list = []
    pids_list = []

    for i, item in enumerate(batch):
        t = item["data"].shape[0]

        # Copy data and dates
        padded_data[i, :t] = item["data"]
        padded_dates[i, :t] = item["dates"]

        # Mark which positions are NOT padded
        pad_mask[i, :t] = False

        masks_list.append(item["masks"])
        pids_list.append(item["pid"])

    masks = torch.stack(masks_list, dim=0)  # (B, H, W)
    pids = torch.stack(pids_list, dim=0)  # (B,)

    return {
        "data": padded_data,
        "dates": padded_dates,
        "masks": masks,
        "pid": pids,
        "pad_mask": pad_mask,
    }
