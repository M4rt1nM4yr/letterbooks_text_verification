import random
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from src.data.utils.alphabet import Alphabet
from src.data.utils.subsequent_mask import subsequent_mask
from src.utils.constants import *
from src.data.utils.custom_collate import collate, slim_collate


def generate_dummy_data(
        alphabet=None,
        batch_size = 10,
        img_channels = 1,
        img_height = 64,
        max_img_width = 1600,
        max_tgt_len = 55,
        cuda = False,
):
    if alphabet is None:
        alphabet = Alphabet(
            characters="abcdefghijklmnopqrstuvwxyz",
        )
    x = torch.randn(batch_size, img_channels, img_height, random.randint(500,max_img_width))
    widths = torch.randint(150, x.shape[-1], size=(batch_size,))
    y = torch.randint(0, len(alphabet.toPosition), size=(batch_size, max_tgt_len))
    y_widths = torch.randint(5, max_tgt_len, size=(batch_size,))
    y_s2s = torch.zeros((batch_size, y.shape[1] + 2), dtype=torch.long)
    tgt_key_padding_mask = torch.zeros(y_s2s.shape[0], y_s2s.shape[1]).int()
    for idx, w in enumerate(y_widths):
        tgt_key_padding_mask[idx, int(w)+2:] = 1
        y_s2s[idx] = torch.cat([
            torch.tensor([alphabet.toPosition[START_OF_SEQUENCE]]),
            y[idx],
            torch.tensor([alphabet.toPosition[END_OF_SEQUENCE]]),
        ])
        y_s2s[idx, int(w)+2:] = alphabet.toPosition[PAD]
        y[idx, int(w):] = alphabet.toPosition[PAD]
    tgt_key_padding_mask = tgt_key_padding_mask.bool()
    tgt_mask = subsequent_mask(y_s2s.shape[-1])

    if cuda:
        x = x.cuda()
        y = y.cuda()
        y_s2s = y_s2s.cuda()
        widths = widths.cuda()
        tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        tgt_mask = tgt_mask.cuda()

    return {
        LINE_IMAGE: x,
        UNPADDED_IMAGE_WIDTH: widths,
        TEXT_LOGITS_CTC: y,
        TEXT_LOGITS_S2S: y_s2s,
        UNPADDED_TEXT_LEN: y_widths,
        TGT_MASK: tgt_mask,
        TGT_KEY_PADDING_MASK: tgt_key_padding_mask,
    }

class DummyDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            alphabet=None,
            batch_size=10,
            img_channels=1,
            img_height=64,
            max_img_width=1600,
            max_tgt_len=55,
            dataset_size = 100,
            collate_fn="default",
    ):
        super().__init__()
        if alphabet is None:
            alphabet = Alphabet(
                characters="abcdefghijklmnopqrstuvwxyz",
            )
        self.batch_size = batch_size
        self.n_workers = 8
        self.train_data = DummyDataset(
            alphabet=alphabet,
            img_channels=img_channels,
            img_height=img_height,
            max_img_width=max_img_width,
            max_tgt_len=max_tgt_len,
            dataset_size=dataset_size,
        )
        self.val_data = DummyDataset(
            alphabet=alphabet,
            img_channels=img_channels,
            img_height=img_height,
            max_img_width=max_img_width,
            max_tgt_len=max_tgt_len,
            dataset_size=dataset_size,
        )
        self.test_data = DummyDataset(
            alphabet=alphabet,
            img_channels=img_channels,
            img_height=img_height,
            max_img_width=max_img_width,
            max_tgt_len=max_tgt_len,
            dataset_size=dataset_size,
        )

        if collate_fn == "default":
            self.collate_fn = collate
        elif collate_fn == "slim":
            self.collate_fn = slim_collate

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.n_workers,
        )

class DummyDataset(Dataset):
    def __init__(
            self,
            alphabet=None,
            img_channels=1,
            img_height=64,
            max_img_width=1600,
            max_tgt_len=55,
            dataset_size = 100,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        if alphabet is None:
            alphabet = Alphabet(
                characters="abcdefghijklmnopqrstuvwxyz",
            )
        self.alphabet = alphabet
        self.img_channels = img_channels
        self.img_height = img_height
        self.max_img_width = max_img_width
        self.max_tgt_len = max_tgt_len

    def __getitem__(self, item):
        x = torch.randn(self.img_channels, self.img_height, random.randint(150, self.max_img_width))
        y = torch.randint(0, len(self.alphabet.toPosition), size=(random.randint(4,self.max_tgt_len),))

        return {
            LINE_IMAGE: x,
            TEXT: self.alphabet.logits_to_string(y),
            TEXT_DIPLOMATIC: self.alphabet.logits_to_string(y),
            TEXT_LOGITS_CTC: y,
            TEXT_LOGITS_S2S: torch.cat([
                    torch.tensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                    y,
                    torch.tensor([self.alphabet.toPosition[END_OF_SEQUENCE]]),
                ]),
            TEXT_DIPLOMATIC_LOGITS_CTC: y,
            TEXT_DIPLOMATIC_LOGITS_S2S: torch.cat([
                torch.LongTensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                y,
                torch.LongTensor([self.alphabet.toPosition[END_OF_SEQUENCE]]),
            ]),
            WRITER: torch.randint(0,100, size=(1,)),
            "name": f"dummy{random.randint(0,1000)}",
        }

    def __len__(self):
        return self.dataset_size