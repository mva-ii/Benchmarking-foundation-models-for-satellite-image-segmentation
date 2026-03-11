from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from fm_benchmark_remote_sensing.data.preview.pastis_dataset import PASTIS_Dataset


@dataclass(frozen=True)
class PastisDatasetItem:
    pastis_target: Any
    pastis_data: Any
    ae_embedding: torch.Tensor
    tessera_embedding: torch.Tensor


class PastisEmbeddingPreviewDataset(Dataset):
    """
    Fetch everything from the PASTIS dataset,
    and the corresponding embeddings from the AE and Tessera models.
    """

    def __init__(
        self,
        pastis_r_root: Path,
        ae_root: Path,
        tessera_root: Path,
    ) -> None:
        self.pastis_dataset = PASTIS_Dataset(folder=pastis_r_root)
        self.ae_root = ae_root
        self.tessera_root = tessera_root

    def __len__(self) -> int:
        return len(self.pastis_dataset)

    def __getitem__(self, idx: int) -> PastisDatasetItem:
        patch_id = self.pastis_dataset.id_patches[idx]
        data, target = self.pastis_dataset[idx]
        return PastisDatasetItem(
            pastis_target=target,
            pastis_data=data,
            ae_embedding=torch.from_numpy(np.load(self.ae_root / f"{patch_id}.npy")),
            tessera_embedding=torch.from_numpy(
                np.load(self.tessera_root / f"{patch_id}.npy")
            ),
        )
