from typing import Optional, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision import transforms
import wandb

from src.utils.constants import *


class VizS2SHTRPredictionCallback(Callback):
    def __init__(
            self,
            max_samples=4,
            every_n_epochs=10,
    ):
        super().__init__()
        self.max_samples = max_samples
        self.every_n_epochs = every_n_epochs
        self.toPIL = transforms.ToPILImage()

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot visualize reconstructions (callback) with `Trainer` that has no logger."
            )
        if batch_idx > 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return

        x = batch[LINE_IMAGE]

        for i in range(min(self.max_samples, len(x))):
            tgt_text = batch[TEXT][i]
            tgt_text_diplomatic = batch[TEXT_DIPLOMATIC][i]
            # pred_text = pl_module.pred_text_s2s[i] if hasattr(pl_module, "pred_text_s2s") else ""
            # pred_text_cheat = pl_module.pred_text_s2s_cheat[i] if hasattr(pl_module, "pred_text_s2s_cheat") else ""
            for l in trainer.loggers:
                if isinstance(l, WandbLogger):
                    l.experiment.log({
                        f"Train prediction sample {i}": [wandb.Image(
                            self.toPIL(x[i]),
                            caption=f"TGT: {tgt_text} | TGT DIPL: {tgt_text_diplomatic}"
                        )]
                    })

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot visualize reconstructions (callback) with `Trainer` that has no logger."
            )
        if batch_idx > 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return

        x = batch[LINE_IMAGE]

        for i in range(min(self.max_samples, len(x))):
            tgt_text = batch[TEXT][i]
            tgt_text_diplomatic = batch[TEXT_DIPLOMATIC][i]
            pred_text = pl_module.val_pred_text_s2s[i] if hasattr(pl_module, "val_pred_text_s2s") else ""
            pred_text_cheat = pl_module.val_pred_text_s2s_cheat[i] if hasattr(pl_module, "val_pred_text_s2s_cheat") else ""
            for l in trainer.loggers:
                if isinstance(l, WandbLogger):
                    l.experiment.log({
                        f"Val prediction sample {i}": [wandb.Image(
                            self.toPIL(x[i]),
                            caption=f"TGT: {tgt_text} | TGT DIPL: {tgt_text_diplomatic} | PRED: {pred_text} | PRED CHEAT: {pred_text_cheat}"
                        )]
                    })

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot visualize reconstructions (callback) with `Trainer` that has no logger."
            )
        if batch_idx > 0:
            return

        x = batch[LINE_IMAGE]

        for i in range(min(self.max_samples, len(x))):
            tgt_text = batch[TEXT][i]
            tgt_text_diplomatic = batch[TEXT_DIPLOMATIC][i] if pl_module.transcription_target == "dipl" else batch[TEXT][i]
            pred_text = pl_module.test_pred_text_s2s[i] if hasattr(pl_module, "test_pred_text_s2s") else ""
            pred_text_cheat = pl_module.test_pred_text_s2s_cheat[i] if hasattr(pl_module, "test_pred_text_s2s_cheat") else ""
            for l in trainer.loggers:
                if isinstance(l, WandbLogger):
                    l.experiment.log({
                        f"Test prediction sample {i}": [wandb.Image(
                            self.toPIL(x[i]),
                            caption=f"TGT: {tgt_text} | TGT DIPL ({pl_module.transcription_target}): {tgt_text_diplomatic} | PRED: {pred_text} | PRED CHEAT: {pred_text_cheat}"
                        )]
                    })