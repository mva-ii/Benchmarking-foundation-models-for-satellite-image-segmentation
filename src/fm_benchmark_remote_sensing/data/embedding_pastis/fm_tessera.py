from __future__ import annotations

from pathlib import Path

from .fm_base import FMBase, Layout


class TesseraFM(FMBase):
    """
    Embeddings : .../pastis_embedding/TESSERA_EMBEDDING/<pid>.npy
    Layout : (H,W,D)
    """

    def __init__(self, embedding_root: Path) -> None:
        super().__init__()
        self.emb_root = embedding_root

    def _path_for_pid(self, pid: int) -> Path:
        path = self.emb_root / f"{pid}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing Tessera embedding for pid={pid}: {path}")
        return path

    def _layout(self) -> Layout:
        return "HWD"
