import os
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob
from PIL import Image
from pytorch_lightning.core import LightningModule
from torchmetrics.classification import MultilabelAUROC
import CARZero.builder as builder
import CARZero
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from nltk.tokenize import RegexpTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import Linear as LoRALinear
import re
from typing import List, Tuple, Dict, Set, Optional, Sequence, Callable
import random
from finetune.utils import generate_mcq

class NIHBiMCQDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()
        
        choices, answer_idx = generate_mcq(labels, self.class_names, no_hyb=self.cfg.data.text.no_hyb)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"],
            "attention_mask" : tokens["attention_mask"],
            "token_type_ids" : tokens["token_type_ids"],
            "cap_len" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }
        
class NIHDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()

        return {
            "imgs": img,
            "label" : torch.tensor(labels, dtype=torch.float),
        }
        
class NIHPosNegDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()
        
        if self.cfg.data.text.neg :
            choices, answer_idx = generate_mcq2(labels, self.class_names, no_hyb=True)
            choices = choices[answer_idx]
        else :
            if labels[-1] == 1 :
                choices = f"There is no finding."
            else :
                pos_indices = [i for i, v in enumerate(labels[:-1]) if v == 1]
                chosen_idx = random.choice(pos_indices)
                chosen_class = self.class_names[chosen_idx]
                choices = f"There is {chosen_class.replace('_', ' ')}."
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        
        
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )
        
        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"][0],
            "attention_mask" : tokens["attention_mask"][0],
            "token_type_ids" : tokens["token_type_ids"][0],
            "cap_len" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
        }