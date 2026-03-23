"""
Main script for semantic experiments - PyTorch Lightning version (Optimized)
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
Modified for: Class 0 & 19 ignoring & Lightning best practices
"""
import argparse
import json
import os
import pickle as pkl
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src import utils, model_utils
from src.dataset import PASTIS_Dataset
from src.learning.miou import IoU
from src.learning.weight_init import weight_init

# Classes ignorées dans la loss ET les métriques
IGNORE_CLASSES = [0, 19]  # 0=Background, 19=Void label

# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="utae", type=str)
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]", type=str)
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

parser.add_argument("--dataset_folder", default="", type=str)
parser.add_argument("--res_dir", default="./results", help="Path for results")
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--rdm_seed", default=1, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--display_step", default=50, type=int)
parser.add_argument("--cache", dest="cache", action="store_true")
parser.set_defaults(cache=False)

parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument("--val_every", default=1, type=int)

# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class SemanticSegmentationModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        # --- Loss : poids nuls sur Background (0) et Void (19) ---
        # On utilise weight= pour les deux classes plutôt que ignore_index
        # car ignore_index n'accepte qu'un seul entier dans PyTorch.
        weights = torch.ones(config.num_classes).float()
        for cls in IGNORE_CLASSES:
            weights[cls] = 0.0
        self.register_buffer("loss_weights", weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weights)

        # --- Métriques : ignore les mêmes classes ---
        self.train_iou = IoU(num_classes=config.num_classes, ignore_index=IGNORE_CLASSES, cm_device=config.device)
        self.val_iou   = IoU(num_classes=config.num_classes, ignore_index=IGNORE_CLASSES, cm_device=config.device)
        self.test_iou  = IoU(num_classes=config.num_classes, ignore_index=IGNORE_CLASSES, cm_device=config.device)

        self.test_conf_mat = None

    def forward(self, x, dates):
        return self.model(x, batch_positions=dates)

    def _shared_step(self, batch):
        (x, dates), y = batch
        y = y.long()
        out = self(x, dates)
        loss = self.criterion(out, y)
        pred = out.argmax(dim=1)
        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, pred, y = self._shared_step(batch)
        self.train_iou.add(pred, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        miou, acc = self.train_iou.get_miou_acc()
        self.log("train/miou", miou)
        self.log("train/acc", acc)
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._shared_step(batch)
        self.val_iou.add(pred, y)
        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        miou, acc = self.val_iou.get_miou_acc()
        self.log("val/miou", miou, prog_bar=True)
        self.log("val/acc", acc)
        print(f"\n[Epoch {self.current_epoch}] Val mIoU: {miou:.4f} | Acc: {acc:.2f}")
        self.val_iou.reset()

    def test_step(self, batch, batch_idx):
        loss, pred, y = self._shared_step(batch)
        self.test_iou.add(pred, y)
        self.log("test/loss", loss)

    def on_test_epoch_end(self):
        miou, acc = self.test_iou.get_miou_acc()
        self.log("test/miou", miou)
        self.log("test/acc", acc)
        self.test_conf_mat = self.test_iou.conf_metric.value()
        print(f"\n[TEST FINAL] mIoU: {miou:.4f} | Acc: {acc:.2f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config):
    L.seed_everything(config.rdm_seed)
    os.makedirs(config.res_dir, exist_ok=True)

    dt_args = dict(
        folder=config.dataset_folder, norm=True, reference_date=config.ref_date,
        mono_date=config.mono_date, target="semantic", sats=["S2"],
    )

    # Split fixe : train=1,2,3 | val=5 | test=4
    dt_train = PASTIS_Dataset(**dt_args, folds=[1, 2, 3], cache=config.cache)
    dt_val   = PASTIS_Dataset(**dt_args, folds=[5],       cache=config.cache)
    dt_test  = PASTIS_Dataset(**dt_args, folds=[4])

    print("Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test)))

    collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)

    train_loader = data.DataLoader(
        dt_train, batch_size=config.batch_size, shuffle=True,
        drop_last=True, num_workers=config.num_workers, collate_fn=collate_fn,
    )
    val_loader = data.DataLoader(
        dt_val, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn,
    )
    test_loader = data.DataLoader(
        dt_test, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn,
    )

    # Modèle
    model = model_utils.get_model(config, mode="semantic")
    model.apply(weight_init)
    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS:", config.N_params)

    with open(os.path.join(config.res_dir, "conf.json"), "w") as f:
        json.dump(vars(config), f, indent=4)

    lit_model = SemanticSegmentationModule(model, config)

    # Callbacks & Logger
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.res_dir, filename="best_model",
        monitor="val/miou", mode="max", save_top_k=1,
    )
    logger = TensorBoardLogger(save_dir=config.res_dir, name="logs")

    # Trainer
    accelerator = "cpu" if config.device == "cpu" else "gpu"
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb],
        check_val_every_n_epoch=config.val_every,
        log_every_n_steps=config.display_step,
        num_sanity_val_steps=0,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Test sur le meilleur checkpoint
    print("\nTesting best epoch . . .")
    trainer.test(lit_model, dataloaders=test_loader, ckpt_path="best")

    # Sauvegarde finale
    conf_mat = lit_model.test_conf_mat
    if torch.is_tensor(conf_mat):
        conf_mat = conf_mat.cpu().numpy()

    metrics = {
        "test_miou": float(trainer.callback_metrics.get("test/miou", float("nan"))),
        "test_acc":  float(trainer.callback_metrics.get("test/acc",  float("nan"))),
        "test_loss": float(trainer.callback_metrics.get("test/loss", float("nan"))),
    }
    print("Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
        metrics["test_loss"], metrics["test_acc"], metrics["test_miou"]
    ))

    with open(os.path.join(config.res_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config.res_dir, "conf_mat.pkl"), "wb"))


if __name__ == "__main__":
    config = parser.parse_args()

    # Parsing sécurisé des listes
    for arg in ["encoder_widths", "decoder_widths", "out_conv"]:
        val = getattr(config, arg)
        if isinstance(val, str):
            setattr(config, arg, json.loads(val))

    assert config.num_classes == config.out_conv[-1]
    pprint.pprint(vars(config))
    main(config)
