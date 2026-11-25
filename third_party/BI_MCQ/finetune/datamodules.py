import os
from torch.utils.data import DataLoader
import torch
from PIL import Image

from finetune.datasets import NIHBiMCQDataset, NIHDataset, NIHPosNegDataset
import CARZero.builder as builder
import CARZero
import pytorch_lightning as pl

class NIHBiMCQDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        print("Using NIHBiMCQDataset")
        if self.cfg.data.fewshot.enabled :
            fewshot_ratio = self.cfg.data.fewshot.ratio
            train_size = len(self.train_df)
            fewshot_size = int(train_size * fewshot_ratio)
            self.train_df = self.train_df.sample(n=fewshot_size, random_state=42).reset_index(drop=True)
            print(f"Few-shot enabled: Using {fewshot_size} samples out of {train_size} for training.")
        self.train_dataset = NIHBiMCQDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHBiMCQDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHBiMCQDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )
        
        
class NIHDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        self.train_dataset = NIHDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

class NIHPosNegDataModule(pl.LightningDataModule):
    def __init__(self, cfg, root, train_df, val_df, test_df):
        super().__init__()
        self.cfg = cfg
        self.root = root
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_transform = builder.build_transformation(cfg, 'train')
        self.test_transform = builder.build_transformation(cfg, 'test')

    def setup(self, stage=None):
        print("Using NIHPosNegDataset")
        self.train_dataset = NIHPosNegDataset(self.train_df, self.cfg, transform=self.train_transform)
        self.val_dataset = NIHPosNegDataset(self.val_df, self.cfg, transform=self.test_transform)
        self.test_dataset = NIHPosNegDataset(self.test_df, self.cfg, transform=self.test_transform)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.test.num_workers,
            pin_memory=True
        )