from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from my_code.models.fm_base import FMBase


def read_patch_ids(metadata_geojson: Path) -> List[int]:
    """
    Lit metadata.geojson et récupère les ID_PATCH.
    """
    metadata = json.loads(metadata_geojson.read_text(encoding="utf-8"))
    feats = metadata.get("features", [])

    pids: List[int] = []
    for ft in feats:
        props = ft.get("properties", {})
        if "ID_PATCH" not in props:
            raise RuntimeError(f"Pas d'ID_PATCH dans {metadata_geojson}")
        pids.append(int(props["ID_PATCH"]))


    pids.sort()
    return pids


def target_path(pastis_root: Path, pid: int) -> Path:
    """
    PASTIS-R/ANNOTATIONS/TARGET_<pid>.npy
    """
    path = pastis_root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Pas de target pour l'identifiant{pid}: {path}")
    return path


def load_mask_hw(path: Path) -> torch.Tensor:
    """
    Charge TARGET_<pid>.npy et retourne un masque (H, W) int64.

    Format de target :
      - (3, H, W) avec :
          canal 0 : labels (0..19)
          canal 1 : info auxiliaire 
          canal 2 : info auxiliaire
    """
    t = np.load(path)
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"TARGET doit être de shape (3,H,W), reçu {t.shape} pour {path}")

    labels = t[0] 

    if labels.min() < 0 or labels.max()>19 :
        raise ValueError(f"Labels incorrects dans {path}: min={labels.min()}, mxn={labels.max()}")

    return torch.from_numpy(labels.astype(np.int64, copy=False))



class PastisEmbeddingDataset(Dataset):
    """
    Un item =
        embeddings: (H,W,D) float32
      + masks:      (H,W)   int64
      + pid:                int64 
    """
    def __init__(
        self,
        pastis_root: Path,
        fm: FMBase,
        subset_patch_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.pastis_root = Path(pastis_root)
        self.fm = fm

        pids = read_patch_ids(self.pastis_root / "metadata.geojson")

        if subset_patch_ids is not None:
            wanted = {int(x) for x in subset_patch_ids}
            pids = [pid for pid in pids if pid in wanted]

        if not pids:
            raise RuntimeError("Dataset vide après filtrage.")

        self.pids = pids

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = int(self.pids[idx])

        fm_out = self.fm.load(pid)
        emb = fm_out.embeddings_hwd  # (H,W,D)

        mpath = target_path(self.pastis_root, pid)
        mask = load_mask_hw(mpath)  # (H,W)

        if tuple(emb.shape[:2]) != tuple(mask.shape):
            raise ValueError(
                f"Décorrellation des dim pour {pid}: emb HW={tuple(emb.shape[:2])} alors que mask HW={tuple(mask.shape)}"
            )

        return {
            "embeddings": emb,
            "masks": mask,
            "pid": torch.tensor(pid, dtype=torch.int64),
        }
