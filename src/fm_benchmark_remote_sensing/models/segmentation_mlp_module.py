from __future__ import annotations

from typing import Any, Tuple, cast

import lightning as L
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

from fm_benchmark_remote_sensing.data import PASTIS_LABEL_NAMES
from fm_benchmark_remote_sensing.models.mlp_head import MLPHeadConfig, PixelMLPHead
import wandb


class SegmentationMLPModule(L.LightningModule):
    """
    - logits = head(embeddings)
    - Train/Val : mIoU & F1 macro loggés par époque (courbes)
    - Test : utilisé uniquement quand on lance trainer.test(...), pour produire confmat_final.pt

    Adaptation labels :
      - Labels d'origine attendus dans {0..19}
      - Classes à ignorer : 0 et 19
      - Remapping interne :
          0,19 -> ignore_index
          1..18 -> 0..17  (décalage -1)
      => la head doit donc prédire num_classes=18
    """

    def __init__(
        self,
        head_cfg: MLPHeadConfig,
        remap_to_ignore_index: int,
        ignore_labels: Tuple[int, int] = (0, 19),
    ) -> None:
        super().__init__()
        self.head = PixelMLPHead(head_cfg)

        # Valeur unique d'ignore pour la loss/metrics (doit être hors [0..num_classes-1])
        self.ignore_index = int(remap_to_ignore_index)

        # Labels d'origine à ignorer (avant remapping)
        self.ignore_labels = tuple(int(x) for x in ignore_labels)

        # Nombre de classes "apprises" (après remapping)
        self.num_classes = int(head_cfg.num_classes)

        if 0 <= self.ignore_index < self.num_classes:
            raise ValueError(
                f"ignore_index={self.ignore_index} must be outside [0..{self.num_classes - 1}] "
                "(reserved for ignored pixels)."
            )

        # On retire les labels ignorés et on applique le même décalage que `_remap_targets`.
        remapped: dict[int, str] = {}
        for orig_lab, orig_name in PASTIS_LABEL_NAMES.items():
            if int(orig_lab) in self.ignore_labels:
                continue
            new_lab = int(orig_lab) - 1
            if 0 <= new_lab < self.num_classes:
                remapped[new_lab] = str(orig_name)

        missing = [i for i in range(self.num_classes) if i not in remapped]
        if missing:
            expected = len(PASTIS_LABEL_NAMES) - len(self.ignore_labels)
            raise ValueError(
                "REMAPPED_LABEL incomplete; missing remapped indices: "
                + ", ".join(map(str, missing))
                + f". This usually means head_cfg.num_classes={self.num_classes} doesn't match the "
                + f"remapping scheme (expected {expected} learned classes for ignore_labels={self.ignore_labels})."
            )

        remapped[self.ignore_index] = "ignore"
        self.REMAPPED_LABEL: dict[int, str] = remapped
        # Noms des classes apprises (0..num_classes-1). Ne pas inclure "ignore" ici.
        self.REMAPPED_CLASS_NAMES: list[str] = [
            self.REMAPPED_LABEL[i] for i in range(self.num_classes)
        ]

        # Loss : CrossEntropy attend logits(B,K,H,W) et target(B,H,W)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # --- Train metrics ---
        self.train_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.train_f1_macro = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )

        # --- Val metrics ---
        self.miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.f1_macro = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )

        # --- Sauvegarde de prédictions ---
        self._saved_pred_batches: list[dict[str, torch.Tensor]] = []
        self.table_data: list[list[Any]] = []
        self.save_n_batches = (
            3  # nombre de batchs à sauvegarder (uniquement à la dernière époque)
        )

        # --- Test accumulators (for W&B confusion matrix) ---
        self.test_preds: list[torch.Tensor] = []
        self.test_y_true: list[torch.Tensor] = []

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)  # (B,H,W,K)

    def _remap_targets(self, masks_bhw: torch.Tensor) -> torch.Tensor:
        """
        Remappe les labels d'origine -> labels pour training.

        Hypothèse : labels d'origine dans {0..19}.
        But :
          - ignorer 0 et 19
          - garder 1..18 mais les rendre contigus 0..17

        Règles :
          - mask == 0  -> ignore_index
          - mask == 19 -> ignore_index
          - 1..18      -> (mask - 1) dans 0..17
        """
        m = masks_bhw.clone()

        # Marque d'abord les labels à ignorer
        for lab in self.ignore_labels:
            m[m == lab] = self.ignore_index

        # Décalage pour rendre les classes contiguës.
        # Important : ne décaler que les pixels non ignorés.
        keep = m != self.ignore_index
        m[keep] = m[keep] - 1

        return m

    def _loss(self, logits_bhwk: torch.Tensor, masks_bhw: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss attend logits(B,K,H,W) et target(B,H,W)
        logits_bkhw = logits_bhwk.permute(0, 3, 1, 2).contiguous()
        return self.criterion(logits_bkhw, masks_bhw)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask_orig = batch["masks"]

        mask = self._remap_targets(mask_orig)
        logits = self(emb)
        loss = self._loss(logits, mask)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        preds = torch.argmax(logits, dim=-1)
        self.train_miou.update(preds, mask)
        self.train_f1_macro.update(preds, mask)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train/mIoU_epoch",
            self.train_miou,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/F1_macro_epoch",
            self.train_f1_macro,
            prog_bar=True,
            sync_dist=True,
        )
        if self.trainer is not None and self.trainer.is_global_zero:
            miou = self.trainer.callback_metrics.get("train/mIoU_epoch")
            f1 = self.trainer.callback_metrics.get("train/F1_macro_epoch")
            if miou is not None:
                self.print("[TRAIN] mIoU =", float(miou))
            if f1 is not None:
                self.print("[TRAIN] F1_macro =", float(f1))

        self.train_miou.reset()
        self.train_f1_macro.reset()

    def shared_test_step(
        self, batch, batch_idx: int, stage: str
    ) -> dict[str, torch.Tensor]:
        emb = batch["embeddings"]
        mask_orig = batch["masks"]
        # Remapping labels
        mask = self._remap_targets(mask_orig)
        logits = self(emb)
        preds = torch.argmax(logits, dim=-1)
        self.miou.update(preds, mask)
        self.f1_macro.update(preds, mask)
        loss = self._loss(logits, mask)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {
            "loss": loss,
            "pid": batch["pid"],
            "embeddings": emb,
            "targets": mask,
            "preds": preds,
            "logits": logits,
        }

    def validation_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.shared_test_step(batch, batch_idx, stage="val")

    def on_validation_start(self) -> None:
        self.table_data.clear()

    def shared_on_validation_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        assert self.trainer.max_epochs is not None
        if not self.trainer.is_global_zero:
            return
        if not isinstance(outputs, dict):
            return
        for i in range(batch["pid"].shape[0]):
            black_image = torch.zeros(
                (3, outputs["targets"].shape[1], outputs["targets"].shape[2]),
                dtype=torch.uint8,
            )
            mask_img = wandb.Image(
                black_image,
                masks={
                    "predictions": {
                        "mask_data": outputs["preds"][i].detach().cpu(),
                        "class_labels": self.REMAPPED_LABEL,
                    },
                    "ground_truth": {
                        "mask_data": outputs["targets"][i].detach().cpu(),
                        "class_labels": self.REMAPPED_LABEL,
                    },
                },
            )
            self.table_data.append(
                [
                    outputs["pid"][i].item(),
                    mask_img,
                    outputs["preds"][i].cpu().numpy(),
                    outputs["targets"][i].cpu().numpy(),
                    outputs["logits"][i].cpu().numpy(),
                    outputs["embeddings"][i].cpu().numpy(),
                ]
            )

    def on_validation_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.shared_on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def shared_epoch_end(self, stage: str) -> None:
        self.log(
            f"{stage}/mIoU_epoch",
            self.miou,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/F1_macro_epoch",
            self.f1_macro,
            prog_bar=True,
            sync_dist=True,
        )

        if self.trainer is not None and self.trainer.is_global_zero:
            miou = self.trainer.callback_metrics.get(f"{stage}/mIoU_epoch")
            f1 = self.trainer.callback_metrics.get(f"{stage}/F1_macro_epoch")
            if miou is not None:
                self.print(f"[{stage.upper()}] mIoU =", float(miou))
            if f1 is not None:
                self.print(f"[{stage.upper()}] F1_macro =", float(f1))

        self.miou.reset()
        self.f1_macro.reset()
        assert self.trainer.max_epochs is not None

        if self.current_epoch == self.trainer.max_epochs - 1:
            logger = cast(WandbLogger, self.logger)
            logger.log_table(
                f"{stage}_predictions_final_epoch",
                columns=[
                    "patch_id",
                    "comparison_image",
                    "prediction_mask",
                    "ground_truth_mask",
                    "logits",
                    "embedding",
                ],
                data=self.table_data,
            )

    def on_validation_epoch_end(self) -> None:
        self.shared_epoch_end(stage="val")

    def on_test_start(self) -> None:
        self.test_preds.clear()
        self.test_y_true.clear()
        self.table_data.clear()

    def test_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self.shared_test_step(batch, batch_idx, stage="test")
        self.test_preds.append(outputs["preds"])
        self.test_y_true.append(outputs["targets"])
        return outputs

    def on_test_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.shared_on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self) -> None:
        self.shared_epoch_end(stage="test")
        flat_preds: list[int] = []
        flat_true: list[int] = []
        for preds_bhw, true_bhw in zip(self.test_preds, self.test_y_true):
            preds_flat = preds_bhw.reshape(-1)
            true_flat = true_bhw.reshape(-1)
            keep = true_flat != self.ignore_index
            flat_preds.extend(preds_flat[keep].to(torch.int64).tolist())
            flat_true.extend(true_flat[keep].to(torch.int64).tolist())

        wandb.log(
            {
                "confmat_test": wandb.plot.confusion_matrix(
                    preds=flat_preds,
                    y_true=flat_true,
                    class_names=self.REMAPPED_CLASS_NAMES,
                )
            }
        )
