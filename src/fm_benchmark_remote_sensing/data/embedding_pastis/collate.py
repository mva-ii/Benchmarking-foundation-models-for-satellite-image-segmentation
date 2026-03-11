from __future__ import annotations

from typing import Dict, List

import torch


def collate_items(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Assemble un batch à partir d'items envoyés par PastisEmbeddingDataset.

    Entrée : Liste de B Item
    Dans chaque Item:
      - "embeddings": (H,W,D) float32
      - "masks":      (H,W)   int64
      - "pid":        (1)     int64

    Batch renvoyé:
      - "embeddings": (B,H,W,D)
      - "masks":      (B,H,W)
      - "pid":        (B)
    """
    if not items:
        raise ValueError("Empty batch in collate_items().")

    embs = [it["embeddings"] for it in items]
    masks = [it["masks"] for it in items]
    pids = [it["pid"] for it in items]

    shape0 = tuple(embs[0].shape)
    for e in embs:
        if tuple(e.shape) != shape0:
            raise ValueError(f"Incohérence dans les tailles des embeddings {tuple(e.shape)} / {shape0}")

    mshape0 = tuple(masks[0].shape)
    for m in masks:
        if tuple(m.shape) != mshape0:
            raise ValueError(f"Incohérence dans les tailles des masks{tuple(m.shape)} / {mshape0}")

    batch = {
        "embeddings": torch.stack(embs, dim=0),
        "masks": torch.stack(masks, dim=0),
        "pid": torch.stack(pids, dim=0),
    }
    return batch
