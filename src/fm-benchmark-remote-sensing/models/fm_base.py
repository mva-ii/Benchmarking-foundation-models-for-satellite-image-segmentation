from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

Layout = Literal["HWD", "DHW"]


@dataclass(frozen=True)
class FMOutput:
    """
    Sortie standardisée pour le pipeline.
    """
    embeddings_hwd: torch.Tensor  # de la forme (H, W, D) float32
    embedding_dim: int            # D


class FMBase:
    """
    But unique : cadrer le chargement d'un embedding à partir d'un patch_id (pid).
    """
    def __init__(self) -> None:
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    def load(self, pid: int) -> FMOutput:
        """
        Charge un embedding (H,W,D) float32.
        """
        path = self._path_for_pid(pid)
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"Embedding de taille : {arr.shape} à {path}, on attend 3 dimensions.")

        emb = self._to_hwd(arr, self._layout()) # normalisation en (H,W,D)
        d = int(emb.shape[-1])

        if self._embedding_dim is None:
            self._embedding_dim = d
        elif self._embedding_dim != d:
            raise ValueError(
                f"Taille d'embedding incohérente : cas référence : {self._embedding_dim}, actuel : {d} (ID:{pid}, Path:{path})"
            )

        return FMOutput(embeddings_hwd=emb, embedding_dim=d)

    def _to_hwd(self, arr: np.ndarray, layout: Layout) -> torch.Tensor:
        """
        assure float32 (H,W,D).
        """
        if layout == "HWD":
            out = arr
        elif layout == "DHW":
            out = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f"Layout non reconnu: {layout}")

        out = out.astype(np.float32, copy=False)
        return torch.from_numpy(out)

    # Fonctions à implémenter dans les sous-classes
    def _path_for_pid(self, pid: int) -> Path:
        raise NotImplementedError

    def _layout(self) -> Layout:
        raise NotImplementedError
