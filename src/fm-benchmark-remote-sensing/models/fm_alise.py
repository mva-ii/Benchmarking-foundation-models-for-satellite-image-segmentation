from __future__ import annotations

from pathlib import Path

from .fm_base import FMBase, Layout


class AliseFM(FMBase):
    """
    Embeddings : .../pastis_embedding/ALISE_EMB/alise_embedding_<pid>.npy
    Layout : (D,H,W) -> converti dans fm_base en (H,W,D)
    """
    def __init__(self, pastis_root: Path) -> None:
        super().__init__()
        self.emb_root = pastis_root.parent / "pastis_embedding" / "ALISE_EMB"

    def _path_for_pid(self, pid: int) -> Path:
        path = self.emb_root / f"alise_embedding_{pid}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Pas d'embedding alise_embedding_{pid}.npy en {path}")
        return path

    def _layout(self) -> Layout:
        return "DHW"
