from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightning as L
from torch.utils.data import DataLoader

from my_code.models.fm_base import FMBase
from .pastis_r_embedding_dataset import PastisEmbeddingDataset
from .collate import collate_items


def read_pid_to_fold(metadata_geojson: Path) -> Dict[int, int]:
    """
    Retourne dict(pid : Fold) ("Fold" récupéré dans metadata.geojson compris entre 1 et 5).
    """
    metadata = json.loads(metadata_geojson.read_text(encoding="utf-8"))
    feats = metadata.get("features", [])

    pid_to_fold: Dict[int, int] = {}
    for ft in feats:
        props = ft.get("properties", {})
        if "ID_PATCH" not in props:
            continue
        if "Fold" not in props:
            raise RuntimeError(f'Pas de "Fold" pour pid={props.get("ID_PATCH")} dans {metadata_geojson}')

        pid_to_fold[int(props["ID_PATCH"])] = int(props["Fold"])

    if not pid_to_fold:
        raise RuntimeError(f"Aucune correspondance pid/Fold dans{metadata_geojson}")

    return pid_to_fold


def split_by_folds(pid_to_fold: Dict[int, int], val_fold: int, test_fold: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits disjoint:
      - test => test_fold
      - val  => val_fold
      - train => le reste
    """
    val_fold = int(val_fold)
    test_fold = int(test_fold)
    if val_fold == test_fold:
        raise ValueError(f"val_fold et test_fold sont identiques : {val_fold}")

    train_pids: List[int] = []
    val_pids: List[int] = []
    test_pids: List[int] = []

    for pid, fold in pid_to_fold.items():
        if fold == test_fold:
            test_pids.append(pid)
        elif fold == val_fold:
            val_pids.append(pid)
        else:
            train_pids.append(pid)

    train_pids.sort()
    val_pids.sort()
    test_pids.sort()

    if not train_pids or not val_pids or not test_pids:
        raise RuntimeError(
            f"Split vide: len_train={len(train_pids)}, len_val={len(val_pids)}, len_test={len(test_pids)} "
            f"avec : (val_fold={val_fold}, test_fold={test_fold})"
        )

    return train_pids, val_pids, test_pids


class EmbeddingDataModule(L.LightningDataModule):
    """
    DataModule simple:
      - split par Fold dans metadata.geojson
      - 3 dataloaders: train/val/test
      - collate basique (collate_items)
    """

    def __init__(
        self,
        pastis_root: str,
        fm: FMBase,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        val_fold: int,
        test_fold: int,
        subset_patch_ids: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.pastis_root = Path(pastis_root)
        self.fm = fm
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.val_fold = int(val_fold)
        self.test_fold = int(test_fold)
        self.subset_patch_ids = subset_patch_ids

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None) -> None:
        pid_to_fold = read_pid_to_fold(self.pastis_root / "metadata.geojson")
        train_pids, val_pids, test_pids = split_by_folds(pid_to_fold, self.val_fold, self.test_fold)

        if self.subset_patch_ids is not None:
            wanted = {int(x) for x in self.subset_patch_ids}
            train_pids = [pid for pid in train_pids if pid in wanted]
            val_pids = [pid for pid in val_pids if pid in wanted]
            test_pids = [pid for pid in test_pids if pid in wanted]

        if stage in (None, "fit"):
            self.train_ds = PastisEmbeddingDataset(self.pastis_root, self.fm, subset_patch_ids=train_pids)
            self.val_ds = PastisEmbeddingDataset(self.pastis_root, self.fm, subset_patch_ids=val_pids)

        if stage in (None, "test"):
            self.test_ds = PastisEmbeddingDataset(self.pastis_root, self.fm, subset_patch_ids=test_pids)

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("setup() n'a pas créé le dataset correspondant pour l'entrainement")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_items,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds is None. setup('fit') must run before val_dataloader().")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_items,
        )



    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("setup() n'a pas créé le dataset correspondant pour le test")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_items,
        )
