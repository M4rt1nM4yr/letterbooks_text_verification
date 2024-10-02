from typing import Optional, Any
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.exceptions import MisconfigurationException

class ValueTrackerCallback(Callback):
    def __init__(
            self,
            metrics,
    ):
        super().__init__()
        self.metrics = metrics

    def on_validation_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use value tracker (callback) with `Trainer` that has no logger."
            )

        for metric in self.metrics:
            metric_value = getattr(pl_module, metric, None)
            for l in trainer.loggers:
                l.log_metrics({metric: metric_value}, step=trainer.current_epoch,)