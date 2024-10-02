import time
import os
import shutil
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.data.dipl_dataset import DiplDataset
from src.data.sample_augmentation.ocrodeg import OcrodegAug
from src.data.utils.noTransform import NoTransform
from src.data.utils.custom_collate import collate
from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

class DiplDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root,
            dataset_type="nbb_dipl",
            batch_size=8,
            n_workers=8,
            other_charset = None,
            invert_charset_order=False,
            line_height=64,
            train_samples_per_epoch=-1,
            books_train=["Band2"],
            books_val=["Band2"],
            books_test=["Band2"],
            sample_augmentation=None,
            max_samples = [-1,-1,-1],
            new_root="",
            abbreviations=False,
            **kwargs
    ):
        super().__init__()
        if len(new_root.split("/")) > 1:
            new_root = new_root.rstrip("/") + time.time().__str__()
            print(f"Copying data to {new_root}")
            shutil.copytree(root, new_root)
            root = new_root
        self.batch_size = batch_size
        self.n_workers = n_workers
        if sample_augmentation is None:
            sample_augmentation = OcrodegAug()
        self.train_data = DiplDataset(
            dataset_type=dataset_type,
            other_charset=other_charset,
            root=root,
            split="train",
            line_height=line_height,
            sample_augmentation=sample_augmentation,
            train_samples_per_epoch=train_samples_per_epoch,
            max_samples=max_samples[0],
            books=books_train,
            invert_charset_order=invert_charset_order,
            abbreviations=abbreviations,
            **kwargs)
        self.val_data = DiplDataset(
            dataset_type=dataset_type,
            other_charset=other_charset,
            root=root,
            split="val",
            line_height=line_height,
            sample_augmentation=NoTransform(),
            max_samples=max_samples[1],
            books=books_val,
            invert_charset_order=invert_charset_order,
            abbreviations=abbreviations,
            **kwargs)
        self.test_data = DiplDataset(
            dataset_type=dataset_type,
            other_charset=other_charset,
            root=root,
            split="test",
            line_height=line_height,
            sample_augmentation=NoTransform(),
            max_samples=max_samples[2],
            books=books_test,
            invert_charset_order=invert_charset_order,
            abbreviations=abbreviations,
            **kwargs)
        log.info(f"Train set size: {len(self.train_data)} \n"
                 f"Validation set size: {len(self.val_data)} \n"
                 f"Test set size: {len(self.test_data)}")
        if len(new_root.split("/")) > 1:
            shutil.rmtree(new_root)
            print(f"removed folder {new_root}")

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_data, collate_fn=collate, batch_size=self.batch_size,
            num_workers=self.n_workers, shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, collate_fn=collate, batch_size=self.batch_size,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, collate_fn=collate, batch_size=self.batch_size,
            num_workers=self.n_workers,
        )
