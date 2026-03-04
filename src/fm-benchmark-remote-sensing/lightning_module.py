from __future__ import annotations

from pathlib import Path
from typing import Tuple

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

from my_code.models.mlp_head import MLPHeadConfig, PixelMLPHead


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
        lr: float,
        ignore_index: int,
        ignore_labels: Tuple[int, int] = (0, 19),
    ) -> None:
        super().__init__()
        self.head = PixelMLPHead(head_cfg)
        self.lr = float(lr)

        # Valeur unique d'ignore pour la loss/metrics (doit être hors [0..num_classes-1])
        self.ignore_index = int(ignore_index)

        # Labels d'origine à ignorer (avant remapping)
        self.ignore_labels = tuple(int(x) for x in ignore_labels)

        # Nombre de classes "apprises" (après remapping)
        self.num_classes = int(head_cfg.num_classes)

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
        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.val_f1_macro = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )

        # --- Test (confusion matrix) ---
        self.test_confmat = MulticlassConfusionMatrix(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        # --- Sauvegarde de prédictions ---
        self._saved_pred_batches = []
        self.save_n_batches = 3  # nombre de batchs à sauvegarder (uniquement à la dernière époque)

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
        keep = (m != self.ignore_index)
        m[keep] = m[keep] - 1

        return m

    def _loss(self, logits_bhwk: torch.Tensor, masks_bhw: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss attend logits(B,K,H,W) et target(B,H,W)
        logits_bkhw = logits_bhwk.permute(0, 3, 1, 2).contiguous()
        return self.criterion(logits_bkhw, masks_bhw)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask_orig = batch["masks"]

        # Remapping labels
        mask = self._remap_targets(mask_orig)

        logits = self(emb)
        loss = self._loss(logits, mask)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = torch.argmax(logits, dim=-1)
        self.train_miou.update(preds, mask)
        self.train_f1_macro.update(preds, mask)

        return loss

    def on_train_epoch_end(self) -> None:
        miou = self.train_miou.compute()
        f1 = self.train_f1_macro.compute()

        self.log("train/mIoU", miou, prog_bar=True, sync_dist=True)
        self.log("train/F1_macro", f1, prog_bar=True, sync_dist=True)

        self.train_miou.reset()
        self.train_f1_macro.reset()

    def on_after_backward(self) -> None:
        # Debug possible : vérifier que la head reçoit bien des gradients au tout début.
        if self.global_step == 0:
            has_grad = False
            max_grad = 0.0
            for p in self.head.parameters():
                if p.grad is not None:
                    has_grad = True
                    max_grad = max(max_grad, float(p.grad.abs().max()))
            # self.print(f"[DEBUG] head_has_grad={has_grad} head_max_abs_grad={max_grad}")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask_orig = batch["masks"]

        # Remapping labels
        mask = self._remap_targets(mask_orig)

        logits = self(emb)

        preds = torch.argmax(logits, dim=-1)
        self.val_miou.update(preds, mask)
        self.val_f1_macro.update(preds, mask)

        loss = self._loss(logits, mask)

        # Sauvegarde uniquement à la dernière époque, sur quelques batchs
        if (
            self.current_epoch == self.trainer.max_epochs - 1
            and len(self._saved_pred_batches) < self.save_n_batches
        ):
            pid = batch["pid"]
            self._saved_pred_batches.append(
                {
                    "pid": pid.detach().cpu() if isinstance(pid, torch.Tensor) else pid,
                    "embeddings": emb.detach().cpu(),
                    # Sauvegarde des deux versions pour debug : originale + remappée
                    "targets_orig": mask_orig.detach().cpu(),
                    "targets": mask.detach().cpu(),
                    "logits": logits.detach().cpu(),
                }
            )

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        miou = self.val_miou.compute()
        f1 = self.val_f1_macro.compute()

        self.log("val/mIoU", miou, prog_bar=True, sync_dist=True)
        self.log("val/F1_macro", f1, prog_bar=True, sync_dist=True)

        if self.trainer is not None and self.trainer.is_global_zero:
            self.print("[VAL] mIoU =", float(miou))
            self.print("[VAL] F1_macro =", float(f1))

        self.val_miou.reset()
        self.val_f1_macro.reset()

        # On ne sauvegarde qu'à la dernière époque
        if self.current_epoch != self.trainer.max_epochs - 1:
            return
        if not self._saved_pred_batches:
            return

        save_dir = Path(self.trainer.logger.log_dir) / "saved_predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

        for data in self._saved_pred_batches:
            pid = data["pid"]

            # Normalisation: construire une liste de pid (1 ou B éléments)
            if isinstance(pid, int):
                pid_list = [pid]
            elif isinstance(pid, torch.Tensor):
                pid = pid.detach().cpu()
                if pid.ndim == 0:
                    pid_list = [int(pid.item())]
                else:
                    pid_list = [int(x) for x in pid.flatten().tolist()]
            else:
                pid_list = [int(pid)]

            # Cas 1: un seul pid pour tout le batch -> sauvegarde telle quelle
            if len(pid_list) == 1:
                out_path = save_dir / f"pred_batch_{pid_list[0]}.pt"
                torch.save(data, out_path)
                continue

            # Cas 2: un pid par élément du batch -> 1 fichier par pid
            emb = data["embeddings"]        # (B,H,W,D) ou équivalent
            tgt_orig = data["targets_orig"] # (B,H,W)
            tgt = data["targets"]           # (B,H,W) remappé
            log = data["logits"]            # (B,H,W,K)

            for j, pid_j in enumerate(pid_list):
                out_path = save_dir / f"pred_batch_{pid_j}.pt"
                torch.save(
                    {
                        "pid": torch.tensor(pid_j),
                        "embeddings": emb[j].clone(),
                        "targets_orig": tgt_orig[j].clone(),
                        "targets": tgt[j].clone(),
                        "logits": log[j].clone(),
                    },
                    out_path,
                )

        self._saved_pred_batches.clear()

    def test_step(self, batch, batch_idx: int) -> None:
        emb = batch["embeddings"]
        mask_orig = batch["masks"]

        # Remapping labels
        mask = self._remap_targets(mask_orig)

        logits = self(emb)
        preds = torch.argmax(logits, dim=-1)
        self.test_confmat.update(preds, mask)

    def on_test_epoch_end(self) -> None:
        confmat = self.test_confmat.compute()

        if self.trainer is not None and self.trainer.is_global_zero:
            out_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path("logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "confmat_final.pt"
            torch.save(confmat.detach().cpu(), out_path)

            self.print("Confmat finale sauvegardée dans:", str(out_path))
            self.print("La voici :")
            self.print(confmat)

        self.test_confmat.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
