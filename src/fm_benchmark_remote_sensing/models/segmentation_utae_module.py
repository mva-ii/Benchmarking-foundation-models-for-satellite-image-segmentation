"""Lightning module for UTAE temporal segmentation model."""

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
from fm_benchmark_remote_sensing.models.utae.utae import UTAE
import wandb


class SegmentationUTAEModule(L.LightningModule):
    """Lightning module for UTAE temporal segmentation.

    - Takes temporal sequences (B, T, C, H, W)
    - Returns segmentation logits (B, num_classes, H, W) (channel-first)
    - Logs: mIoU, F1 macro, accuracy per epoch
    - Handles label remapping: 0 and 19 ignored, 1-18 -> 0-17
    """

    def __init__(
        self,
        input_dim: int = 10,
        encoder_widths: list[int] | None = None,
        decoder_widths: list[int] | None = None,
        out_conv: list[int] | None = None,
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        pad_value: float = 0.0,
        padding_mode: str = "reflect",
        remap_to_ignore_index: int = 18,
        save_n_batches: int | None = None,
    ) -> None:
        """
        Args:
            input_dim: Number of input channels (default 10 for S2)
            encoder_widths: Encoder channel widths
            decoder_widths: Decoder channel widths
            out_conv: Output conv layers
            str_conv_k: Strided conv kernel size
            str_conv_s: Strided conv stride
            str_conv_p: Strided conv padding
            agg_mode: Temporal aggregation mode
            encoder_norm: Encoder normalization type
            n_head: Number of attention heads
            d_model: Attention model dimension
            d_k: Key-query dimension
            pad_value: Padding value for temporal sequences
            padding_mode: Spatial padding mode
            remap_to_ignore_index: Index for ignored labels
            save_n_batches: Number of batches to save for W&B
        """
        super().__init__()

        self.input_dim = input_dim
        self.pad_value = pad_value
        self.ignore_index = int(remap_to_ignore_index)
        self.num_classes = 18  # After remapping: 1-18 -> 0-17

        # Default values for UTAE parameters
        if encoder_widths is None:
            encoder_widths = [64, 64, 64, 128]
        if decoder_widths is None:
            decoder_widths = [32, 32, 64, 128]
        if out_conv is None:
            out_conv = [32, self.num_classes]
        elif int(out_conv[-1]) != self.num_classes:
            raise ValueError(
                f"out_conv[-1]={out_conv[-1]} must equal num_classes={self.num_classes} "
                f"(learned classes after remapping). Update your config (e.g. out_conv: [32, {self.num_classes}])."
            )

        # Initialize UTAE model
        self.utae = UTAE(
            input_dim=input_dim,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=out_conv,
            str_conv_k=str_conv_k,
            str_conv_s=str_conv_s,
            str_conv_p=str_conv_p,
            agg_mode=agg_mode,
            encoder_norm=encoder_norm,
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            encoder=False,
            return_maps=False,
            pad_value=pad_value,
            padding_mode=padding_mode,
        )

        if 0 <= self.ignore_index < self.num_classes:
            raise ValueError(
                f"ignore_index={self.ignore_index} must be outside [0..{self.num_classes - 1}]"
            )

        self.save_n_batches = save_n_batches

        # Build remapped label names
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
                f"REMAPPED_LABEL incomplete; missing: {missing}. "
                f"Expected 18 learned classes for ignore_labels=(0, 19)."
            )

        remapped[self.ignore_index] = "ignore"
        self.REMAPPED_LABEL: dict[int, str] = remapped
        self.REMAPPED_CLASS_NAMES: list[str] = [
            self.REMAPPED_LABEL[i] for i in range(self.num_classes)
        ]

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Training metrics
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

        # Validation/Test metrics
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

        # Prediction storage
        self.table_data: list[list[Any]] = []
        self._saved_batch_count = 0

        # Test accumulators
        self.test_preds: list[torch.Tensor] = []
        self.test_y_true: list[torch.Tensor] = []

    def forward(
        self, data: torch.Tensor, dates: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through UTAE.

        Args:
            data: (B, T, C, H, W) temporal sequences
            dates: (B, T) day offsets for positional encoding (optional)

        Returns:
            logits: (B, num_classes, H, W)
        """
        # UTAE expects (B, T, C, H, W) and returns channel-first logits (B, out_conv[-1], H, W)
        logits = self.utae(data, batch_positions=dates)
        return logits

    def _remap_targets(self, masks_bhw: torch.Tensor) -> torch.Tensor:
        """
        Remap original labels (0-19) to training labels (0-17).

        - Ignore 0 (Background) and 19 (Void) -> ignore_index
        - 1-18 -> 0-17
        """
        ignore_mask = (masks_bhw == BKG_LABEL_INDEX) | (masks_bhw == VOID_LABEL_INDEX)
        m = masks_bhw.clone()
        m[~ignore_mask] = m[~ignore_mask] - 1
        m[ignore_mask] = self.ignore_index
        return m

    def _as_bkhw(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalize logits to (B, K, H, W)."""
        if logits.ndim != 4:
            raise ValueError(f"Expected 4D logits, got shape={tuple(logits.shape)}")

        # UTAE returns channel-first logits by default.
        if logits.shape[1] == self.num_classes:
            return logits.contiguous()

        # Be tolerant to channel-last logits if a caller wraps/reorders outputs.
        if logits.shape[-1] == self.num_classes:
            return logits.permute(0, 3, 1, 2).contiguous()

        raise ValueError(
            f"Logits must have num_classes={self.num_classes} either in dim=1 (B,K,H,W) "
            f"or dim=-1 (B,H,W,K). Got shape={tuple(logits.shape)}"
        )

    def _preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices with shape (B, H, W)."""
        if logits.ndim != 4:
            raise ValueError(f"Expected 4D logits, got shape={tuple(logits.shape)}")

        if logits.shape[1] == self.num_classes:  # (B, K, H, W)
            return torch.argmax(logits, dim=1)
        if logits.shape[-1] == self.num_classes:  # (B, H, W, K)
            return torch.argmax(logits, dim=-1)

        raise ValueError(
            f"Logits must have num_classes={self.num_classes} either in dim=1 or dim=-1. "
            f"Got shape={tuple(logits.shape)}"
        )

    def _loss(self, logits: torch.Tensor, masks_bhw: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        logits_bkhw = self._as_bkhw(logits)
        return self.criterion(logits_bkhw, masks_bhw)

    @override
    def training_step(self, batch, _: int) -> torch.Tensor:
        data = batch["data"]  # (B, T, C, H, W)
        dates = batch["dates"]  # (B, T)
        mask_orig = batch["masks"]  # (B, H, W)

        mask = self._remap_targets(mask_orig)
        logits = self(data, dates)  # (B, num_classes, H, W)
        loss = self._loss(logits, mask)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        preds = self._preds_from_logits(logits)
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
        data = batch["data"]
        dates = batch["dates"]
        mask_orig = batch["masks"]

        mask = self._remap_targets(mask_orig)
        logits = self(data, dates)
        preds = self._preds_from_logits(logits)

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
            "data": data,
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

        # Log confusion matrix
        wandb.log(
            {
                "confmat_test": wandb.plot.confusion_matrix(
                    preds=flat_preds,
                    y_true=flat_true,
                    class_names=self.REMAPPED_CLASS_NAMES,
                )
            }
        )
