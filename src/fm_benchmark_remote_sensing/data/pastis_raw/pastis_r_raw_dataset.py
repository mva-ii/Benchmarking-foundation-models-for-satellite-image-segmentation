"""Raw PASTIS-R dataset with CSV-based patch filtering for scarce data support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import Dataset

from fm_benchmark_remote_sensing.data.preview.pastis_dataset import PASTIS_Dataset


def read_patch_ids_from_csv(csv_path: Path) -> List[int]:
    """
    Read patch IDs from a CSV file.
    The CSV is expected to have an 'id_patch' column.

    Follows the same pattern as ALISE.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df_selected_patch = pd.read_csv(csv_path)

    if "id_patch" not in df_selected_patch.columns:
        raise ValueError(f"CSV file {csv_path} does not have 'id_patch' column")

    patch_ids = df_selected_patch["id_patch"].astype("int").tolist()

    if not patch_ids:
        raise RuntimeError(f"No patch IDs found in {csv_path}")

    return patch_ids


class PastisRawDataset(Dataset):
    """
    Wraps PASTIS_Dataset with optional CSV-based patch filtering for scarce data.

    Returns tuples of: {data, dates, masks, pid}
    where:
      - data: (T, C, H, W) time-series tensor
      - dates: (T,) day offsets tensor
      - masks: (H, W) segmentation labels
      - pid: patch ID (scalar)
    """

    def __init__(
        self,
        pastis_r_root: Path,
        sats: Optional[List[str]] = None,
        reference_date: str = "2018-09-01",
        csv_file: Optional[Path] = None,
        folds: Optional[List[int]] = None,
        norm: bool = True,
    ):
        """
        Args:
            pastis_r_root: Path to PASTIS_R directory
            sats: List of satellites to load (default ["S2"])
            reference_date: Reference date for date encoding
            csv_file: Path to CSV with patch IDs to include (for scarce data)
            folds: List of folds to include (for fold-based splitting)
            norm: Whether to normalize data
        """
        if sats is None:
            sats = ["S2"]

        self.pastis_dataset = PASTIS_Dataset(
            folder=str(pastis_r_root),
            sats=sats,
            reference_date=reference_date,
            norm=norm,
            folds=folds,
        )

        # Optional patch filtering via CSV
        if csv_file is not None:
            self.patch_ids = read_patch_ids_from_csv(csv_file)
        else:
            self.patch_ids = list(self.pastis_dataset.id_patches)

        # Create mapping from our index to PASTIS dataset index
        self.idx_to_pastis_idx = []
        for patch_id in self.patch_ids:
            try:
                pastis_idx = list(self.pastis_dataset.id_patches).index(patch_id)
                self.idx_to_pastis_idx.append(pastis_idx)
            except ValueError:
                # Patch ID not in dataset, skip it
                continue

        self.patch_ids = [
            self.pastis_dataset.id_patches[i] for i in self.idx_to_pastis_idx
        ]

    def __len__(self) -> int:
        return len(self.idx_to_pastis_idx)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
              - "data": (T, C, H, W) time-series
              - "dates": (T,) day offsets
              - "masks": (H, W) semantic labels
              - "pid": patch ID (scalar tensor)
        """
        pastis_idx = self.idx_to_pastis_idx[idx]
        pid = self.patch_ids[idx]

        # Get data from PASTIS dataset
        (data_dict, dates_dict), target = self.pastis_dataset[pastis_idx]

        # Concatenate satellite data along channel dimension
        # data_dict is {satellite: (T, C, H, W)}
        data_list = []
        dates_list = []

        for sat in self.pastis_dataset.sats:
            sat_data = data_dict[sat]  # (T, C, H, W)
            sat_dates = dates_dict[sat]  # (T,)
            data_list.append(sat_data)
            if not dates_list:  # Use first satellite's dates
                dates_list = sat_dates

        # Stack along channel dimension
        data = torch.cat(data_list, dim=1)  # (T, C_total, H, W)
        dates = torch.from_numpy(dates_list).float()  # (T,)
        masks = torch.from_numpy(target).long()  # (H, W)

        if target.ndim == 3:  # target is (C, H, W)
            masks = masks[0]  # Take first channel (semantic labels)

        return {
            "data": data,
            "dates": dates,
            "masks": masks,
            "pid": torch.tensor(pid, dtype=torch.long),
        }
