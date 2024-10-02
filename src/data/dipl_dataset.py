import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.utils.fixed_line_height import FixedHeightResize
from src.data.utils.foreground_checker import ForegroundChecker
from src.utils.constants import *
from src.data.dataset_fetcher.nbb_github_fetcher import preload_all_nbb_github
from src.data.utils.noTransform import NoTransform
from src.data.utils.alphabet_generation import generate_alphabet

class DiplDataset(Dataset):
    def __init__(
            self,
            root,
            dataset_type="nbb_dipl",
            other_charset = None,
            invert_charset_order=False,
            split="train",
            line_height=64,
            books=["Band2"],
            sample_augmentation=NoTransform(),
            train_samples_per_epoch=-1,
            max_samples = -1,
            abbreviations=False,
            return_plain_image=False,
            return_text="both",
            **kwargs
    ):
        super().__init__()
        self.return_plain_image = return_plain_image
        self.train_samples_per_epoch = train_samples_per_epoch
        self.split = split
        self.alphabet = generate_alphabet(dataset_type, other_charset, invert_charset_order)
        self.sample_names, self.meta_data, self.images = preload_all_nbb_github(
            root=root,
            alphabet=self.alphabet,
            books=books, split=split,
            height=line_height,
            return_text = return_text,
            max_samples=max_samples,
            abbreviations=abbreviations,
            **kwargs
        )
        resize = FixedHeightResize(height=line_height)
        self.transform = transforms.Compose([
            sample_augmentation,
            resize,
            transforms.ToTensor(),
            lambda x: 1 - x,
        ])

    def __getitem__(self, item):
        if self.train_samples_per_epoch > 0 and self.split == "train":
            item = torch.randint(low=0, high=len(self.sample_names), size=(1,)).item()
        name = self.sample_names[item]
        img = self.images[name]
        out_dict =  {
            "name": name,
            LINE_IMAGE: self.transform(img),
            TEXT: self.meta_data[name]["text_basic"],
            TEXT_DIPLOMATIC: self.meta_data[name]["text_dipl"],
            TEXT_LOGITS_CTC: self.meta_data[name]["text_logits_basic"],
            TEXT_LOGITS_S2S: torch.cat([
                torch.LongTensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                self.meta_data[name]["text_logits_basic"],
                torch.LongTensor([self.alphabet.toPosition[END_OF_SEQUENCE]]),
            ]),
            TEXT_DIPLOMATIC_LOGITS_CTC: self.meta_data[name]["text_logits_dipl"],
            TEXT_DIPLOMATIC_LOGITS_S2S: torch.cat([
                torch.LongTensor([self.alphabet.toPosition[START_OF_SEQUENCE]]),
                self.meta_data[name]["text_logits_dipl"],
                torch.LongTensor([self.alphabet.toPosition[END_OF_SEQUENCE]]),
            ]),
            WRITER: self.meta_data[name]["writer"],
        }
        if self.return_plain_image:
            out_dict[PLAIN_IMAGE] = transforms.ToPILImage()(self.transform(img))
        return out_dict

    def __len__(self):
        if self.train_samples_per_epoch > 0:
            return self.train_samples_per_epoch
        return len(self.sample_names)
