from typing import Optional, Any

import torch
import pytorch_lightning as pl
import pandas as pd
import os

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional.text.cer import _cer_update

from src.utils.constants import *

class SavePredictionsCallback(pl.Callback):
    def __init__(self, save_dir, k_epochs=1):
        self.save_dir = save_dir
        self.k_epochs = k_epochs
        os.makedirs(save_dir, exist_ok=True)

    # def on_train_epoch_end(self, trainer, pl_module):
    #     epoch = trainer.current_epoch
    #     if epoch % self.k_epochs == 0:
    #         self.save_predictions(trainer, pl_module, 'train', epoch)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.current_epoch % self.k_epochs != 0:
            return
        names = batch[NAME]
        paragraph_ids = batch[PARAGRAPH_ID]

        self.save_predictions(
            names=names,
            paragraph_ids=paragraph_ids,
            targets=pl_module.val_target_text if hasattr(pl_module, "val_target_text") else None,
            predictions=pl_module.val_pred_text_s2s if hasattr(pl_module, "val_pred_text_s2s") else None,
            targets_ctc=pl_module.val_target_text_ctc if hasattr(pl_module, "val_target_text_ctc") else None,
            predictions_ctc=pl_module.val_pred_text_ctc if hasattr(pl_module, "val_pred_text_ctc") else None,
            stage='val',
            epoch=trainer.current_epoch,
        )

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        names = batch[NAME]
        paragraph_ids = batch[PARAGRAPH_ID]
        self.save_predictions(
            names=names,
            paragraph_ids=paragraph_ids,
            targets=pl_module.test_target_text if hasattr(pl_module, "test_target_text") else None,
            predictions=pl_module.test_pred_text_s2s if hasattr(pl_module, "test_pred_text_s2s") else None,
            targets_ctc=pl_module.test_target_text_ctc if hasattr(pl_module, "test_target_text_ctc") else None,
            predictions_ctc=pl_module.test_pred_text_ctc if hasattr(pl_module, "test_pred_text_ctc") else None,
            stage='test',
            epoch=0,
        )

    def save_predictions(
            self,
            stage,
            epoch,
            names,
            paragraph_ids,
            targets = None,
            predictions = None,
            targets_ctc = None,
            predictions_ctc= None,
    ):
        # compute CER
        cer = list()
        cer_ctc = list()

        if targets is not None:
            for target, prediction in zip(targets, predictions):
                errors, total = _cer_update(prediction, target)
                cer.append(float(errors / total))
        else:
            cer = [-1] * len(names)

        if targets_ctc is not None:
            for target, prediction in zip(targets_ctc, predictions_ctc):
                errors, total = _cer_update(prediction, target)
                cer_ctc.append(errors / total)
        else:
            cer_ctc = [-1] * len(names)

        file_path = os.path.join(self.save_dir, f'{stage}_epoch_{epoch}.csv')
        new_df = pd.DataFrame({
            "set": [stage] * len(names),
            'name': names,
            'paragraph_id': paragraph_ids,
            'target': targets,
            'prediction': predictions,
            'target_ctc': targets_ctc,
            'prediction_ctc': predictions_ctc,
            'CER': cer,
            'CER_ctc': cer_ctc,
        })

        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path, sep='|')
            new_df = pd.concat([old_df, new_df], ignore_index=True)
            new_df.to_csv(file_path, index=False, sep='|')
        else:
            new_df.to_csv(file_path, index=False, sep='|')