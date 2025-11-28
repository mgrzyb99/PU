import warnings

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

warnings.filterwarnings("ignore", ".*pkg_resources is deprecated as an API.*")


class AimSaveConfigCallback(SaveConfigCallback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, save_to_log_dir=False, **kwargs)

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        trainer.logger.log_hyperparams({"config": self.config.as_dict()})


if __name__ == "__main__":
    LightningCLI(
        save_config_callback=AimSaveConfigCallback,
        trainer_defaults={
            "logger": {"class_path": "aim.pytorch_lightning.AimLogger"},
            "enable_checkpointing": False,
            "enable_model_summary": False,
            "log_every_n_steps": 1,
        },
    )
