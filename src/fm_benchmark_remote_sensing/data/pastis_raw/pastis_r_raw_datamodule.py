"""Lightning DataModule for raw PASTIS-R data with temporal sequences and scarce support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as L
from torch.utils.data import DataLoader

from .pastis_r_raw_dataset import PastisRawDataset, read_patch_ids_from_csv
from .collate import collate_temporal_sequences


def read_pid_to_fold(metadata_geojson: Path) -> Dict[int, int]:
    """
    Read fold assignment for each patch ID from metadata.geojson.
    Returns dict(pid : Fold) where Fold is between 1 and 5.
    """
    metadata = json.loads(metadata_geojson.read_text(encoding="utf-8"))
    feats = metadata.get("features", [])

    pid_to_fold: Dict[int, int] = {}
    for ft in feats:
        props = ft.get("properties", {})
        if "ID_PATCH" not in props:
            continue
        if "Fold" not in props:
            raise RuntimeError(
                f'No "Fold" for pid={props.get("ID_PATCH")} in {metadata_geojson}'
            )

        pid_to_fold[int(props["ID_PATCH"])] = int(props["Fold"])

    if not pid_to_fold:
        raise RuntimeError(f"No pid/Fold correspondences found in {metadata_geojson}")

    return pid_to_fold


def split_by_folds(
    pid_to_fold: Dict[int, int], val_fold: int, test_fold: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split patches by fold assignments:
      - test => test_fold
      - val => val_fold
      - train => remaining folds
    """
    val_fold = int(val_fold)
    test_fold = int(test_fold)
    if val_fold == test_fold:
        raise ValueError(f"val_fold and test_fold are identical: {val_fold}")

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
            f"Empty split: len_train={len(train_pids)}, len_val={len(val_pids)}, len_test={len(test_pids)} "
            f"with: (val_fold={val_fold}, test_fold={test_fold})"
        )

    return train_pids, val_pids, test_pids


class PastisRawDataModule(L.LightningDataModule):
    """
    Lightning DataModule for raw PASTIS-R temporal sequences with fold-based splitting
    and scarce data support.

    Supports:
      - Sentinel-2 time series (T, 10, H, W)
      - Fold-based train/val/test splitting
      - Scarce data mode via CSV patch selection (similar to ALISE)
      - Variable-length temporal sequences with padding
    """

    def __init__(
        self,
        pastis_r_root: Path,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        val_fold: int = 4,
        test_fold: int = 5,
        reference_date: str = "2018-09-01",
        scarce_csv_root: Optional[Path] = None,
        scarce_fold_idx: Optional[int] = None,
        scarce_nb_patches: Optional[int] = None,
        norm: bool = True,
    ) -> None:
        """
        Args:
            pastis_r_root: Path to PASTIS_R directory
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory
            val_fold: Fold index for validation (1-5)
            test_fold: Fold index for testing (1-5)
            reference_date: Reference date for date encoding
            scarce_csv_root: Root directory containing scarce data CSVs
            scarce_fold_idx: Seed index for scarce experiments
            scarce_nb_patches: Number of patches per fold (for scarce mode)
            norm: Whether to normalize data
        """
        super().__init__()
        self.pastis_root = Path(pastis_r_root)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.val_fold = int(val_fold)
        self.test_fold = int(test_fold)
        self.reference_date = reference_date
        self.norm = norm

        # Scarce data parameters
        self.scarce_csv_root = Path(scarce_csv_root) if scarce_csv_root else None
        self.scarce_fold_idx = scarce_fold_idx
        self.scarce_nb_patches = scarce_nb_patches

        self.train_ds: Optional[PastisRawDataset] = None
        self.val_ds: Optional[PastisRawDataset] = None
        self.test_ds: Optional[PastisRawDataset] = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for train/val/test."""
        pid_to_fold = read_pid_to_fold(self.pastis_root / "metadata.geojson")

        # Use scarce data if parameters are provided
        if (
            self.scarce_csv_root is not None
            and self.scarce_fold_idx is not None
            and self.scarce_nb_patches is not None
        ):
            # Scarce mode: use N patches per *training fold* (i.e. folds != val_fold and != test_fold)
            all_folds = sorted(set(pid_to_fold.values()))
            train_folds = [
                f for f in all_folds if f not in (self.val_fold, self.test_fold)
            ]
            if not train_folds:
                raise RuntimeError(
                    f"No training folds left with val_fold={self.val_fold} and test_fold={self.test_fold}."
                )

            train_pids: List[int] = []
            for fold in train_folds:
                train_csv_file = (
                    self.scarce_csv_root
                    / f"selected_patches_fold_{fold}_nb_{self.scarce_nb_patches}_seed_{self.scarce_fold_idx}.csv"
                )
                train_pids.extend(read_patch_ids_from_csv(train_csv_file))

            # Keep only pids that belong to the expected folds (defensive) and deduplicate.
            train_pids = sorted(
                {
                    pid
                    for pid in train_pids
                    if int(pid_to_fold.get(int(pid), -1)) in set(train_folds)
                }
            )

            val_csv_file = (
                self.scarce_csv_root
                / f"selected_patches_fold_{self.val_fold}_nb_{self.scarce_nb_patches}_seed_{self.scarce_fold_idx}.csv"
            )
            val_pids = read_patch_ids_from_csv(val_csv_file)
            val_pids = sorted(
                {
                    pid
                    for pid in val_pids
                    if int(pid_to_fold.get(int(pid), -1)) == self.val_fold
                }
            )

            # For test, use all patches from test fold
            test_pids = [
                pid for pid, fold in pid_to_fold.items() if fold == self.test_fold
            ]
            test_pids.sort()
        else:
            # Standard fold-based splitting
            train_pids, val_pids, test_pids = split_by_folds(
                pid_to_fold, self.val_fold, self.test_fold
            )

        if stage in (None, "fit"):
            train_folds = set(pid_to_fold[pid] for pid in train_pids)
            val_fold_set = set(pid_to_fold[pid] for pid in val_pids)

            self.train_ds = PastisRawDataset(
                pastis_r_root=self.pastis_root,
                sats=["S2"],
                reference_date=self.reference_date,
                folds=list(train_folds),
                csv_file=None,  # Will filter by patch IDs manually
                norm=self.norm,
            )
            # Filter to training patch IDs
            self.train_ds.patch_ids = [
                p for p in self.train_ds.patch_ids if p in train_pids
            ]
            self.train_ds.idx_to_pastis_idx = [
                self.train_ds.pastis_dataset.id_patches.tolist().index(pid)
                for pid in self.train_ds.patch_ids
            ]

            self.val_ds = PastisRawDataset(
                pastis_r_root=self.pastis_root,
                sats=["S2"],
                reference_date=self.reference_date,
                folds=list(val_fold_set),
                csv_file=None,
                norm=self.norm,
            )
            # Filter to validation patch IDs
            self.val_ds.patch_ids = [p for p in self.val_ds.patch_ids if p in val_pids]
            self.val_ds.idx_to_pastis_idx = [
                self.val_ds.pastis_dataset.id_patches.tolist().index(pid)
                for pid in self.val_ds.patch_ids
            ]

        if stage in (None, "test"):
            test_fold_set = set(pid_to_fold[pid] for pid in test_pids)
            self.test_ds = PastisRawDataset(
                pastis_r_root=self.pastis_root,
                sats=["S2"],
                reference_date=self.reference_date,
                folds=list(test_fold_set),
                csv_file=None,
                norm=self.norm,
            )
            # Filter to test patch IDs
            self.test_ds.patch_ids = [
                p for p in self.test_ds.patch_ids if p in test_pids
            ]
            self.test_ds.idx_to_pastis_idx = [
                self.test_ds.pastis_dataset.id_patches.tolist().index(pid)
                for pid in self.test_ds.patch_ids
            ]

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("setup() did not create training dataset")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_temporal_sequences,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("setup() did not create validation dataset")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_temporal_sequences,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("setup() did not create test dataset")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_temporal_sequences,
            persistent_workers=self.num_workers > 0,
        )
