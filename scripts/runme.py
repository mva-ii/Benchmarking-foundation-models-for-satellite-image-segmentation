import os
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import wandb
from tqdm import tqdm
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

import fm_benchmark_remote_sensing.data  # pylint: disable=W0611 # noqa: F401 used by the CLI to find datamodules
import fm_benchmark_remote_sensing.models  # pylint: disable=W0611 # noqa: F401 used by the CLI to find models
import tempfile


class CustomSaveConfigCallback(SaveConfigCallback):
    """# from https://github.com/Lightning-AI/pytorch-lightning/issues/19728"""

    # Saves full training configuration
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                cast(WandbLogger, logger).watch(pl_module, log="all")
                config = self.config.as_dict()
                logger.log_hyperparams({"config": config})
        return super().save_config(trainer, pl_module, stage)


class BenchmarkLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--checkpoint_reference")
        parser.add_argument("--dataset_artifact_reference")
        parser.add_argument(
            "--datataset_download_path", enable_path=True, required=True, default="data"
        )

    def before_instantiate_classes(self) -> None:
        api = wandb.Api()
        cmd: str = cast(str, self.subcommand)
        if cmd == "test" or cmd == "predict":
            model_artifact = api.artifact(
                self.config[cmd]["checkpoint_reference"], type="model"
            )
            model_artifact_dir = model_artifact.download()
            self.config[cmd]["ckpt_path"] = os.path.join(
                Path(model_artifact_dir).as_posix(), "model.ckpt"
            )

        if self.config[cmd]["dataset_artifact_reference"] is not None:
            dataset_artifact = api.artifact(
                self.config[cmd]["dataset_artifact_reference"],
                type="dataset",
            )
            download_path: Path = self.config[cmd]["datataset_download_path"]
            if not os.path.exists(download_path):
                dataset_artifact.download(root=download_path)
                for zip_path in download_path.glob("*.zip"):
                    with ZipFile(zip_path) as z:
                        for file in tqdm(
                            z.infolist(), desc=f"Extracting {zip_path.name}"
                        ):
                            z.extract(file, download_path)


def cli_main():

    # Print temp directory for debugging
    print(f"Temporary directory: {tempfile.gettempdir()}")

    BenchmarkLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    wandb.finish()  # really important


if __name__ == "__main__":
    cli_main()
