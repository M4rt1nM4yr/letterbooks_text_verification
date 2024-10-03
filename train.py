import os
import shutil
import hydra
import wandb
import pyrootutils
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
os.environ["WANDB__SERVICE_WAIT"] = "300"

from src.callback.value_tracker_callback import ValueTrackerCallback
from src.callback.s2s_htr_prediction_callback import VizS2SHTRPredictionCallback
from src.callback.save_predictions_callback import SavePredictionsCallback
from src.callback.ema_callback import EMA
from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

@hydra.main(version_base="1.2", config_path=str(root/"configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    os.makedirs(os.path.join(cfg.save_dir,cfg.name,cfg.version), exist_ok=True)
    if cfg.run_local:
        log.info("Wandb is used online")
        wandb.login(key=cfg.wandb_key)
        wandb_logger = WandbLogger(project=cfg.name , version=cfg.version, save_dir=cfg.save_dir,
                                   offline=False, settings=wandb.Settings(start_method="fork"))
    else:
        log.info("Wandb is used offline")
        wandb_logger = WandbLogger(project=cfg.name, version=cfg.version, save_dir=cfg.save_dir,
                                   offline=True, settings=wandb.Settings(start_method="fork"))

    dataModule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    criterium = cfg.monitor_criterium if not cfg.model.fast_val==True else cfg.monitor_criterium+"_cheat"
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        alphabet=dataModule.train_data.alphabet,
        metric=criterium,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.save_dir, cfg.name, cfg.version),
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor=criterium,
        mode="min",
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor=criterium,
        patience=cfg.patience,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    value_tracker = ValueTrackerCallback(
        metrics=["decoder_warm_up_factor"]
    )
    viz_callback = VizS2SHTRPredictionCallback(
        max_samples=4,
        every_n_epochs=10,
    )
    save_predictions_callback = SavePredictionsCallback(
        k_epochs=10,
        save_dir=os.path.join(cfg.save_dir, cfg.name, cfg.version, "predictions")
    )
    callbacks = [checkpoint_callback, early_stopping, lr_monitor, value_tracker, viz_callback, save_predictions_callback]
    if cfg.ema == True:
        ema = EMA(decay=0.99)
        callbacks.append(ema)
    trainer: pl.trainer.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.fit(
        model,
        train_dataloaders=dataModule.train_dataloader(),
        val_dataloaders=dataModule.val_dataloader(),
    )

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    trainer.test(model=model, datamodule=dataModule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()