from __future__ import annotations

from pathlib import Path

from .fm_base import FMBase
from .fm_tessera import TesseraFM
from .fm_alphaearth import AlphaEarthFM
from .fm_alise import AliseFM


def build_fm(name: str, pastis_root: Path) -> FMBase:
    key = name.strip().lower()
    if key == "tessera":
        return TesseraFM(pastis_root)
    if key == "alphaearth":
        return AlphaEarthFM(pastis_root)
    if key == "alise":
        return AliseFM(pastis_root)
    raise ValueError("FM name must be one of: tessera | alphaearth | alise")
