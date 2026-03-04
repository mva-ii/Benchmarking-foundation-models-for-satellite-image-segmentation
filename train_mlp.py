from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from fm_benchmark_remote_sensing.config import load_config
from fm_benchmark_remote_sensing.models import build_fm
from fm_benchmark_remote_sensing.data.pastis_r_embedding_datamodule import (
    EmbeddingDataModule,
)
from fm_benchmark_remote_sensing.models.mlp_head import MLPHeadConfig
from fm_benchmark_remote_sensing.lightning_module import SegmentationMLPModule


def head_in_dim(fm, pastis_root: Path) -> int:
    meta = json.loads((pastis_root / "metadata.geojson").read_text(encoding="utf-8"))
    feats = meta.get("features", [])
    if not feats:
        raise RuntimeError("metadata.geojson n'a pas de features.")
    pid0 = int(feats[0]["properties"]["ID_PATCH"])

    out = fm.load(pid0)
    return int(out.embedding_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",  # Chemin par défaut
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    L.seed_everything(cfg.seed, workers=True)

    pastis_root = Path(cfg.pastis_root)
    fm = build_fm(cfg.name, pastis_root=pastis_root)

    in_dim = head_in_dim(fm, pastis_root)

    datamodule = EmbeddingDataModule(
        pastis_root=str(pastis_root),
        fm=fm,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        val_fold=cfg.val_fold,
        test_fold=cfg.test_fold,
    )

    head_cfg = MLPHeadConfig(
        in_dim=in_dim,
        hidden_dim_1=cfg.hidden_dim_1,
        hidden_dim_2=cfg.hidden_dim_2,
        num_classes=cfg.num_classes,
    )
    model = SegmentationMLPModule(
        head_cfg=head_cfg, lr=cfg.lr, ignore_index=cfg.ignore_index
    )

    logger = CSVLogger(save_dir=cfg.out_dir, name=cfg.experiment_name)

    run_dir = Path(logger.log_dir)
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="best-{epoch:02d}-{val_mIoU:.4f}",
        monitor="val/mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,  # écrit aussi last.ckpt
    )

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[ckpt_cb],
    )

    trainer.fit(model, datamodule=datamodule)

    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(run_dir / "model_final.ckpt")
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_cpu, run_dir / "model_final_state_dict.pt")

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
