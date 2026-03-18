from __future__ import annotations

from typing import Any, cast, override

import lightning as L
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torchmetrics import ClasswiseWrapper
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

from fm_benchmark_remote_sensing.data import (
    PASTIS_LABEL_NAMES,
    BKG_LABEL_INDEX,
    VOID_LABEL_INDEX,
)
from fm_benchmark_remote_sensing.models.mlp_head import MLPHeadConfig, PixelMLPHead
import wandb


class SegmentationMLPModule(L.LightningModule):
    """
    - logits = head(embeddings)
    - Train/Val : mIoU & F1 macro loggés par époque (courbes)
    - Test : utilisé uniquement quand on lance trainer.test(...), pour produire confmat_final.pt

    Adaptation labels :
      - Labels d'origine attendus dans {0..19}
      - Classes ignorées (hardcodées) : 0 (Background) et 19 (Void label)
      - Remapping interne : 1..18 -> 0..17 (décalage -1)
      => la head doit donc prédire num_classes=18
    """

    def __init__(
        self,
        head_cfg: MLPHeadConfig,
        remap_to_ignore_index: int,
        save_n_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.head = PixelMLPHead(head_cfg)

        # Valeur unique d'ignore pour la loss/metrics (doit être hors [0..num_classes-1])
        self.ignore_index = int(remap_to_ignore_index)

        # Nombre de classes "apprises" (après remapping)
        self.num_classes = int(head_cfg.num_classes)

        if 0 <= self.ignore_index < self.num_classes:
            raise ValueError(
                f"ignore_index={self.ignore_index} must be outside [0..{self.num_classes - 1}] "
                "(reserved for ignored pixels)."
            )
        self.save_n_batches = save_n_batches
        # Labels 0 et 19 ignorés, labels 1..18 décalés à 0..17.
        remapped: dict[int, str] = {}
        for orig_lab, orig_name in PASTIS_LABEL_NAMES.items():
            if int(orig_lab) in (BKG_LABEL_INDEX, VOID_LABEL_INDEX):
                continue
            new_lab = int(orig_lab) - 1
            if 0 <= new_lab < self.num_classes:
                remapped[new_lab] = str(orig_name)

        missing = [i for i in range(self.num_classes) if i not in remapped]
        if missing:
            raise ValueError(
                "REMAPPED_LABEL incomplete; missing remapped indices: "
                + ", ".join(map(str, missing))
                + f". This usually means head_cfg.num_classes={self.num_classes} doesn't match the "
                + "remapping scheme (expected 18 learned classes for ignore_labels=(0, 19))."
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
        self.train_accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="micro",
        )

        # --- Val metrics ---
        self.val_test_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.val_test_f1_macro = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.val_test_accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="micro",
        )
        self.val_test_f1_per_class = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="none",
        )
        self.wrapped_val_test_f1_per_class = ClasswiseWrapper(
            self.val_test_f1_per_class,
            prefix="F1_",
            labels=self.REMAPPED_CLASS_NAMES,
        )

        # --- Sauvegarde de prédictions ---
        self.table_data: list[list[Any]] = []
        self._saved_batch_count = 0

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
          - ignorer 0 (Background) et 19 (Void label) -> ignore_index
          - 1..18 -> 0..17 (décalage -1)

        Le masque d'ignore est calculé sur le tenseur ORIGINAL pour éviter
        toute collision (ex : label 18=Sorghum ne doit pas être confondu
        avec ignore_index=18 avant le décalage).
        """
        ignore_mask = (masks_bhw == BKG_LABEL_INDEX) | (masks_bhw == VOID_LABEL_INDEX)
        m = masks_bhw.clone()
        m[~ignore_mask] = m[~ignore_mask] - 1
        m[ignore_mask] = self.ignore_index
        return m

    def _loss(self, logits_bhwk: torch.Tensor, masks_bhw: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss attend logits(B,K,H,W) et target(B,H,W)
        logits_bkhw = logits_bhwk.permute(0, 3, 1, 2).contiguous()
        return self.criterion(logits_bkhw, masks_bhw)

    @override
    def training_step(self, batch, _: int) -> torch.Tensor:
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
        self.train_miou(preds, mask)
        self.train_f1_macro(preds, mask)
        self.train_accuracy(preds, mask)
        self.log(
            "train/mIoU_epoch",
            self.train_miou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/F1_macro_epoch",
            self.train_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/accuracy_epoch",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def shared_test_step(self, batch, _: int, stage: str) -> dict[str, torch.Tensor]:
        emb = batch["embeddings"]
        mask_orig = batch["masks"]
        # Remapping labels
        mask = self._remap_targets(mask_orig)
        logits = self(emb)
        preds = torch.argmax(logits, dim=-1)
        self.val_test_miou(preds, mask)
        self.val_test_f1_macro(preds, mask)
        self.val_test_accuracy(preds, mask)
        val_test_f1_per_class_dict = self.wrapped_val_test_f1_per_class(preds, mask)
        val_test_f1_per_class_dict = {
            f"{stage}/{key}": value for key, value in val_test_f1_per_class_dict.items()
        }
        loss = self._loss(logits, mask)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/mIoU_epoch",
            self.val_test_miou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log_dict(
            val_test_f1_per_class_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,  # ! IMPORTANT https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        )

        self.log(
            f"{stage}/F1_macro_epoch",
            self.val_test_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/accuracy_epoch",
            self.val_test_accuracy,
            on_step=False,
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

    def shared_on_batch_end(self, outputs, batch, _: int, __: int = 0) -> None:
        assert self.trainer.max_epochs is not None
        if not self.trainer.is_global_zero:
            return
        if not isinstance(outputs, dict):
            return
        if not (
            self.trainer.state.stage == "test"
            or self.current_epoch == self.trainer.max_epochs - 1
        ):
            return
        if (
            self.save_n_batches is not None
            and self._saved_batch_count >= self.save_n_batches
        ):
            return

        self._saved_batch_count += 1
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
                    outputs["preds"][i].detach().cpu().numpy(),
                    outputs["targets"][i].detach().cpu().numpy(),
                    outputs["logits"][i].detach().cpu().numpy(),
                    outputs["embeddings"][i].detach().cpu().numpy(),
                ]
            )

    def on_validation_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.shared_on_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def shared_epoch_end(self, stage: str) -> None:
        assert self.trainer.max_epochs is not None

        if not self.trainer.is_global_zero:
            return

        if stage == "test" or self.current_epoch == self.trainer.max_epochs - 1:
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
        self.test_preds.append(outputs["preds"].detach())
        self.test_y_true.append(outputs["targets"].detach())
        return outputs

    def on_test_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.shared_on_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self) -> None:
        if not self.trainer.is_global_zero:
            return

        self.shared_epoch_end(stage="test")

        flat_preds: list[int] = []
        flat_true: list[int] = []

        for preds_bhw, true_bhw in zip(self.test_preds, self.test_y_true):
            preds_flat = preds_bhw.reshape(-1)
            true_flat = true_bhw.reshape(-1)
            keep = true_flat != self.ignore_index
            flat_preds.extend(preds_flat[keep].cpu().tolist())
            flat_true.extend(true_flat[keep].cpu().tolist())

        # Log confusion matrix once after accumulating all predictions
        wandb.log(
            {
                "confmat_test": wandb.plot.confusion_matrix(
                    preds=flat_preds,
                    y_true=flat_true,
                    class_names=self.REMAPPED_CLASS_NAMES,
                )
            }
        )
