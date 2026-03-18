from __future__ import annotations

from pathlib import Path
from .mlp_head import MLPHeadConfig
from .segmentation_mlp_module import SegmentationMLPModule
from .segmentation_utae_module import SegmentationUTAEModule


def build_fm(name: str, pastis_root: Path) -> None:
    raise ValueError("FM name must be one of: tessera | alphaearth | alise")


_all = [MLPHeadConfig, SegmentationMLPModule, SegmentationUTAEModule]
