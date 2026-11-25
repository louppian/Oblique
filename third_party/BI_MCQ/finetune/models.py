import os
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
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
from einops import rearrange
from nltk.tokenize import RegexpTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import Linear as LoRALinear
from torchvision import models as tv_models
import re
from finetune.utils import build_t2i_mcq_batch

import random
from typing import List, Tuple, Optional

class PretrainDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None)
            self.freeze_module()
        # if self.cfg.peft.enabled :
        #     self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        bce_loss, mean_auroc, class_auroc = self.metrics(batch, "val")
        return {
            "val_loss": loss.detach(),
            "val_bce_loss": bce_loss.detach(),
            "mean_auroc": mean_auroc.detach(),
            "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss, mean_auroc, class_auroc = self.metrics(batch, "test")
        return {
            "test_loss": loss.detach(),
            "test_bce_loss": bce_loss.detach(),
            "mean_auroc": mean_auroc.detach(),
            "class_auroc": class_auroc.detach()
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss = self.CARZero_model.calc_loss(
            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls
        )

        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        return loss
    
    def metrics(self, batch, split):
        processes_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.prompts, self.device)
        similarity = CARZero.dqn_shot_classification(
            self.CARZero_model,
            batch["imgs"].to(self.device),
            processes_text,).values
        similarity = torch.tensor(similarity).to(self.device)
        labels = batch["label"].to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        probs = torch.sigmoid(similarity)
        preds = (probs > 0.5).float()
        
        self.auroc_metric.update(probs, labels.int())  # 배치 단위로 누적만
        
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_bce_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{split}_mean_auroc",
            mean_auroc,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        metrics = {f"{split}_auroc_{cls}": class_auroc[i] for i, cls in enumerate(self.dm.train_dataset.class_names)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss, mean_auroc, class_auroc
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val_loss", "val_bce_loss", "val_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val_auroc_")}
            for key in class_metrics:
                class_name = key.replace("val_auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()

class BiMCQCARZeroModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts 

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.CARZero_model.i2t_fusion_module.parameters():
                param.requires_grad = False
            for param in self.CARZero_model.t2i_fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
        self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def i2t_forward(self, batch):
        i2t_cls, t2i_cls = self.CARZero_model.i2t_mcq_forward(batch, i2t_only=self.cfg.model.CARZero.single_path)
        
        logits = (i2t_cls + t2i_cls)/2 if self.cfg.model.CARZero.single_path == False else i2t_cls
        
        targets = batch["answer_idx"].to(self.device)
        
        loss = F.cross_entropy(logits, targets, reduction="mean")
        acc = (logits.argmax(dim=1) == targets).float().mean()
        
        return i2t_cls, t2i_cls, loss, acc
    
    def t2i_forward(self, batch):
        batch = build_t2i_mcq_batch(
            batch,
            self.tokenizer,
            self.prompts,
            self.class_names,
            max_length=self.cfg.data.text.word_num,
            num_negatives=2,
            no_hyb=self.cfg.data.text.no_hyb
            )
        
        if len(batch['imgs'].shape) != 5 :
            self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
            return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        i2t_cls, t2i_cls = self.CARZero_model.t2i_mcq_forward(batch, t2i_only=self.cfg.model.CARZero.single_path)
        
        logits = (i2t_cls + t2i_cls)/2 if self.cfg.model.CARZero.single_path == False else t2i_cls
        
        acc = (logits.argmax(dim=1) == batch["answer_idx"].to(self.device)).float().mean()
        
        loss = F.cross_entropy(logits, batch["answer_idx"].to(self.device), reduction="mean")
        
        return i2t_cls, t2i_cls, loss, acc

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        weight = self.cfg.train.loss_weight
        
        i2t_logits_i2t, t2i_logits_i2t, i2t_loss, i2t_acc = self.i2t_forward(batch)
        i2t_logits_t2i, t2i_logits_t2i, t2i_loss, t2i_acc = self.t2i_forward(batch)

        ce_loss = weight * i2t_loss + (1 - weight) * t2i_loss
        
        self.log_dict({f"{split}/loss": ce_loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc},
                  prog_bar=True, on_epoch=True, sync_dist=True)
                 
        return ce_loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_text(
            self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = CARZero.bimcq_classification(
            self.CARZero_model, imgs, pos_text)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_text(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.bimcq_classification(
            self.CARZero_model, imgs, neg_text) # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.loss_weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)
        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        
class PosNegDQNWOSAMLPGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.CARZero_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']

        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts

    def setup(self, stage=None):
        if self.CARZero_model is None:
            self.CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.dm is None:
            self.dm = self.trainer.datamodule
    
    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.CARZero_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.CARZero_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.CARZero_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.CARZero_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.CARZero_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.CARZero_model.text_encoder.parameters()))
        
        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.CARZero_model.t2i_fusion_module.parameters()))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.CARZero_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.shared_step(batch, "val")
        bce_loss = self.metrics(batch, "val")
        return {
            #"val/loss": loss.detach(),
            "val/bce_loss": bce_loss.detach(),
            # "mean_auroc": mean_auroc.detach(),
            # "class_auroc": class_auroc.detach()
        }
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {
            "test/loss": loss.detach(),
            "test/bce_loss": bce_loss.detach(),
        }

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls = self.CARZero_model(batch)
        loss = self.CARZero_model.calc_loss(
        i2t_cls, t2i_cls
        )

        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        
        return loss
        
    def metrics(self, batch, split):
        imgs   = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        # ---------- Positive-prompt similarity ----------
        pos_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, pos_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs  = torch.sigmoid(pos_logits)

        # ---------- Negative-prompt similarity ----------
        neg_text = self.CARZero_model.process_class_prompts(
            self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = CARZero.dqn_shot_classification(
            self.CARZero_model, imgs, neg_text, mcq=self.cfg.model.CARZero.multi, multi=self.cfg.model.CARZero.multi) # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs  = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)

        # ---------- 지표 집계 ----------
        class_auroc      = self.auroc_metric.compute()
        neg_class_auroc  = self.neg_auroc_metric.compute()
        pos_mean_auroc       = class_auroc.mean()
        neg_mean_auroc   = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc",     mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc",     pos_mean_auroc,     prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        # 클래스별 AUROC도 한꺼번에 로깅
        self.log_dict({f"{split}/auroc_{c}":     class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss
        
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")

            # 기본 손실 및 평균 AUROC 출력
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")
            
            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")
                
    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")
            
            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()



class BiMCQMedKLIP(nn.Module):
    """MedKLIP backbone (ResNet image encoder + BERT text encoder) wired into the
    same i2t/t2i MCQ fusion modules used by the CARZero/BiMCQ backbone, so it can
    train against the same NIHBiMCQDataModule/build_t2i_mcq_batch pipeline."""

    def __init__(self, cfg):
        super(BiMCQMedKLIP, self).__init__()

        self.cfg = cfg
        self.d_model = cfg.model.MedKLIP.d_model

        self.bert_model = self._get_bert_basemodel(cfg.model.text.bert_type)
        self.disease_embedding_layer = nn.Linear(768, self.d_model)
        # Unused in the forward paths below, but part of the original MedKLIP checkpoint's
        # state_dict - kept so strict=True checkpoint loading doesn't fail on unexpected keys.
        self.cl_fc = nn.Linear(self.d_model, 768)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        resnet_dict = {
            "resnet18": tv_models.resnet18(pretrained=False),
            "resnet50": tv_models.resnet50(pretrained=False),
        }
        resnet = resnet_dict[cfg.model.MedKLIP.res_base_model]
        num_ftrs = int(resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, self.d_model)

        self.t2i_fusion_module = builder.build_mcq_fusion_module(cfg, cfg.model.fusion.t2i_average_attn_weights)
        self.i2t_fusion_module = builder.build_mcq_fusion_module(cfg, cfg.model.fusion.i2t_average_attn_weights)

    def _get_bert_basemodel(self, bert_model_name):
        return AutoModel.from_pretrained(bert_model_name)

    def image_encoder(self, xis):
        # patch features
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis)  # (B, C, H, W)
        res_fea = rearrange(res_fea, "b d n1 n2 -> b (n1 n2) d")
        h = rearrange(res_fea, "b n d -> (b n) d")
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x, "(b n) d -> b n d", b=batch_size)  # (B, S, dim)
        g_feat = out_emb.mean(dim=1)  # (B, dim)
        return out_emb, g_feat

    def i2t_mcq_forward(self, x, i2t_only=False):
        img_emb_l, img_emb_g = self.image_encoder(x["imgs"])

        B, N, L = x["caption_ids"].shape
        caption_ids = x["caption_ids"].view(B * N, L)
        attention_mask = x["attention_mask"].view(B * N, L)

        query_embed = self.bert_model(input_ids=caption_ids, attention_mask=attention_mask)  # (B*N, L, D)
        text_emb_l = self.disease_embedding_layer(query_embed.last_hidden_state)
        text_emb_g = self.disease_embedding_layer(query_embed.pooler_output)

        D = text_emb_g.shape[-1]
        text_emb_l = text_emb_l.view(B, N, text_emb_l.size(1), text_emb_l.size(2))
        text_emb_g = text_emb_g.view(B, N, text_emb_g.size(1))

        img_emb_ = torch.cat([img_emb_g.unsqueeze(1), img_emb_l], dim=1)
        text_emb_ = torch.cat([text_emb_g.unsqueeze(2), text_emb_l], dim=2)

        img_emb_g_ = img_emb_g.unsqueeze(1)
        img_emb_g_ = img_emb_g_.unsqueeze(1).expand(B, N, 1, D)
        img_emb_g_ = img_emb_g_.reshape(B * N, 1, D)
        img_emb_g_ = img_emb_g_.permute(1, 0, 2)  # (1, B*N, D)

        text_emb_g_ = text_emb_g.reshape(B * N, 1, D)
        text_emb_g_ = text_emb_g_.permute(1, 0, 2)  # (1, B*N, D)

        B_, S_img, D_ = img_emb_.shape
        _, N_, S_txt, _ = text_emb_.shape

        img_emb_ = img_emb_.permute(1, 0, 2)  # (S_img+1, B, D)
        img_emb_ = img_emb_.unsqueeze(2).expand(S_img, B, N, D)
        img_emb_ = img_emb_.reshape(S_img, B * N, D)
        img_emb_ = img_emb_.transpose(0, 1)  # (B*N, S_img, D)

        text_emb_ = text_emb_.permute(2, 0, 1, 3)  # (S_txt, B, N, D)
        text_emb_ = text_emb_.reshape(S_txt, B * N, D)
        text_emb_ = text_emb_.transpose(0, 1)  # (B*N, S_txt, D)

        if i2t_only:
            t2i_logit = None
        else:
            t2i_logit = self.t2i_fusion_module(img_emb_, text_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
            t2i_logit = t2i_logit.view(B, N)

        i2t_logit = self.i2t_fusion_module(text_emb_, img_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
        i2t_logit = i2t_logit.view(B, N)
        return i2t_logit, t2i_logit

    def t2i_mcq_forward(self, x, t2i_only=False):
        imgs = x["imgs"]
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N, C, H, W)

        img_emb_l, img_emb_g = self.image_encoder(imgs_flat)

        B_t, N_t, L = x["caption_ids"].shape
        assert B_t == B and N_t == N

        caption_ids = x["caption_ids"].reshape(B * N, L)
        attention_mask = x["attention_mask"].reshape(B * N, L)

        query_embed = self.bert_model(input_ids=caption_ids, attention_mask=attention_mask)
        text_emb_l = self.disease_embedding_layer(query_embed.last_hidden_state)
        text_emb_g = self.disease_embedding_layer(query_embed.pooler_output)

        _, S_img, D_img = img_emb_l.shape
        img_emb_l_bn = img_emb_l.view(B, N, S_img, D_img)
        img_emb_g_bn = img_emb_g.view(B, N, D_img)

        _, L_txt, D_txt = text_emb_l.shape
        text_emb_l_bn = text_emb_l.view(B, N, L_txt, D_txt)
        text_emb_g_bn = text_emb_g.view(B, N, D_txt)

        img_emb_l_flat = img_emb_l_bn.permute(0, 1, 3, 2)
        img_emb_l_flat = img_emb_l_flat.view(B * N, D_img, -1)
        img_emb_l_flat = img_emb_l_flat.permute(0, 2, 1)
        img_emb_g_pair = img_emb_g_bn.view(B * N, D_img)

        img_emb_ = torch.cat([img_emb_g_pair.unsqueeze(1), img_emb_l_flat], dim=1)  # (B*N, 1+S_img, D)

        text_emb_l_flat = text_emb_l_bn.view(B * N, L_txt, D_txt)
        text_emb_g_pair = text_emb_g_bn.view(B * N, D_txt)
        text_emb_ = torch.cat([text_emb_g_pair.unsqueeze(1), text_emb_l_flat], dim=1)  # (B*N, 1+L, D)

        img_emb_g_ = img_emb_g_pair.view(B * N, 1, D_img).permute(1, 0, 2)
        text_emb_g_ = text_emb_g_pair.view(B * N, 1, D_txt).permute(1, 0, 2)

        t2i_logit = self.t2i_fusion_module(img_emb_, text_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
        t2i_logit = t2i_logit.view(B, N)

        if t2i_only:
            i2t_logit = None
        else:
            i2t_logit = self.i2t_fusion_module(text_emb_, img_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
            i2t_logit = i2t_logit.view(B, N)

        return i2t_logit, t2i_logit

    def inference(self, images, text, mode="both"):
        B = images.shape[0]
        img_emb_l, img_emb_g = self.image_encoder(images)

        query_embed = self.bert_model(input_ids=text["caption_ids"], attention_mask=text["attention_mask"])
        text_emb_l = self.disease_embedding_layer(query_embed.last_hidden_state)
        text_emb_g = self.disease_embedding_layer(query_embed.pooler_output)

        cls_bs = []
        for i in range(B):
            B_, L, D = text_emb_l.shape
            label_img_emb_l = img_emb_l[i:i + 1]
            label_img_emb_g = img_emb_g[i:i + 1].unsqueeze(1)

            query_emb_g_ = text_emb_g.unsqueeze(1).permute(1, 0, 2)

            attention_mask = text["attention_mask"].bool()
            global_valid = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([global_valid, attention_mask], dim=1)
            attention_mask = torch.logical_not(attention_mask)

            label_emb_ = torch.cat([label_img_emb_g, label_img_emb_l], dim=1)
            label_emb_ = label_emb_.expand(B_, -1, -1)
            query_emb_ = torch.cat([text_emb_g.unsqueeze(1), text_emb_l], dim=1)

            t2i_cls, t2i_attn, t2i_feats = self.t2i_fusion_module(
                label_emb_, query_emb_g_, inside_repeat=False, return_feat=True
            )
            t2i_cls = t2i_cls.squeeze(-1).squeeze(-1)

            label_emb_g__ = label_img_emb_g.expand(-1, B_, -1)
            i2t_cls, i2t_attn, i2t_feats = self.i2t_fusion_module(
                query_emb_, label_emb_g__, inside_repeat=False, attention_mask=attention_mask, return_feat=True
            )
            i2t_cls = i2t_cls.squeeze(-1).squeeze(-1)

            if mode == "i2t":
                cls = i2t_cls
            elif mode == "t2i":
                cls = t2i_cls
            else:
                cls = (i2t_cls + t2i_cls) / 2

            cls_bs.append(cls.unsqueeze(0).detach().cpu())

        cls = torch.cat(cls_bs, dim=0)
        return cls.numpy()

    def process_text(self, text, device):
        if isinstance(text, str):
            text = [text]

        processed_text_tensors = []
        for t in text:
            t = t.replace("\n", " ")
            splitter = re.compile(r"[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []
            for t_sent in captions:
                t_sent = t_sent.replace("��", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t_sent.lower())
                if len(tokens) <= 1:
                    continue
                included_tokens = []
                for tok in tokens:
                    tok = tok.encode("ascii", "ignore").decode("ascii")
                    if len(tok) > 0:
                        included_tokens.append(tok)
                all_sents.append(" ".join(included_tokens))

            t_joined = " ".join(all_sents)
            text_tensors = self.tokenizer(
                t_joined,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack([x["attention_mask"] for x in processed_text_tensors])
        token_type_ids = torch.stack([x["token_type_ids"] for x in processed_text_tensors])

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = [len([w for w in txt if not w.startswith("[")]) for txt in text]

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):
        return {k: self.process_text(v, device) for k, v in class_prompts.items()}


class BiMCQMedKLIPModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.MedKLIP_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']

        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts

    def setup(self, stage=None):
        if self.MedKLIP_model is None:
            self.MedKLIP_model = BiMCQMedKLIP(self.cfg)

            ckpt_path = self.cfg.model.MedKLIP.ckpt_path
            self.print("Load model from checkpoint:", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint["model"]
            new_state_dict = OrderedDict(
                (k[len("module."):] if k.startswith("module.") else k, v)
                for k, v in state_dict.items()
            )
            self.MedKLIP_model.load_state_dict(new_state_dict, strict=False)

            self.freeze_module()
            self.print("MedKLIP model loaded and frozen.")
        if self.dm is None:
            self.dm = self.trainer.datamodule

    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        image_modules = (self.MedKLIP_model.res_features, self.MedKLIP_model.res_l1, self.MedKLIP_model.res_l2)
        if freeze_dict.get("image", False):
            for module in image_modules:
                for param in module.parameters():
                    param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.MedKLIP_model.bert_model.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.MedKLIP_model.i2t_fusion_module.parameters():
                param.requires_grad = False
            for param in self.MedKLIP_model.t2i_fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for module in image_modules for p in module.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.MedKLIP_model.bert_model.parameters()))
        self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.MedKLIP_model.i2t_fusion_module.parameters()))
        self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.MedKLIP_model.t2i_fusion_module.parameters()))

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.MedKLIP_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def i2t_forward(self, batch):
        i2t_cls, t2i_cls = self.MedKLIP_model.i2t_mcq_forward(batch, i2t_only=self.cfg.model.MedKLIP.single_path)

        logits = (i2t_cls + t2i_cls) / 2 if self.cfg.model.MedKLIP.single_path == False else i2t_cls

        targets = batch["answer_idx"].to(self.device)

        loss = F.cross_entropy(logits, targets, reduction="mean")
        acc = (logits.argmax(dim=1) == targets).float().mean()

        return i2t_cls, t2i_cls, loss, acc

    def t2i_forward(self, batch):
        batch = build_t2i_mcq_batch(
            batch,
            self.tokenizer,
            self.prompts,
            self.class_names,
            max_length=self.cfg.data.text.word_num,
            num_negatives=2,
            no_hyb=self.cfg.data.text.no_hyb
            )

        if len(batch['imgs'].shape) != 5 :
            self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
            return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        i2t_cls, t2i_cls = self.MedKLIP_model.t2i_mcq_forward(batch, t2i_only=self.cfg.model.MedKLIP.single_path)

        logits = (i2t_cls + t2i_cls) / 2 if self.cfg.model.MedKLIP.single_path == False else t2i_cls

        acc = (logits.argmax(dim=1) == batch["answer_idx"].to(self.device)).float().mean()

        loss = F.cross_entropy(logits, batch["answer_idx"].to(self.device), reduction="mean")

        return i2t_cls, t2i_cls, loss, acc

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bce_loss = self.metrics(batch, "val")
        return {"val/bce_loss": bce_loss.detach()}

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {"test/loss": loss.detach(), "test/bce_loss": bce_loss.detach()}

    def shared_step(self, batch, split):
        weight = self.cfg.train.loss_weight

        i2t_logits_i2t, t2i_logits_i2t, i2t_loss, i2t_acc = self.i2t_forward(batch)
        i2t_logits_t2i, t2i_logits_t2i, t2i_loss, t2i_acc = self.t2i_forward(batch)

        ce_loss = weight * i2t_loss + (1 - weight) * t2i_loss

        self.log_dict({f"{split}/loss": ce_loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc},
                  prog_bar=True, on_epoch=True, sync_dist=True)

        return ce_loss

    def metrics(self, batch, split):
        imgs = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)

        pos_text = self.MedKLIP_model.process_text(self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = self.MedKLIP_model.inference(imgs, pos_text)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs = torch.sigmoid(pos_logits)

        neg_text = self.MedKLIP_model.process_text(self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = self.MedKLIP_model.inference(imgs, neg_text)  # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:, :-1]).int()

        self.auroc_metric.update(pos_probs, labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        weight = self.cfg.train.loss_weight
        bce_loss = weight * pos_bce_loss + (1 - weight) * neg_bce_loss

        class_auroc = self.auroc_metric.compute()
        neg_class_auroc = self.neg_auroc_metric.compute()
        pos_mean_auroc = class_auroc.mean()
        neg_mean_auroc = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        self.log(f"{split}/bce_loss", bce_loss, prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc", mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc", pos_mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        self.log_dict({f"{split}/auroc_{c}": class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss

    def on_before_optimizer_step(self, optimizer):
        total_norm_sq = sum(
            p.grad.detach().data.norm(2).item() ** 2
            for p in self.MedKLIP_model.parameters() if p.grad is not None
        )
        self.log("grad_norm", total_norm_sq ** 0.5, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")

            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")

    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()




class KADImageEncoder(nn.Module):
    """ResNet50 image encoder matching KAD's ModelRes checkpoint layout exactly
    (keeps the redundant self.resnet alongside self.res_features so state_dict
    keys line up with the pretrained KAD_224/best_valid.pt checkpoint)."""

    def __init__(self, res_base_model="resnet50"):
        super(KADImageEncoder, self).__init__()
        resnet_dict = {"resnet50": tv_models.resnet50(pretrained=True)}
        self.resnet = resnet_dict[res_base_model]
        num_ftrs = int(self.resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, 768)

    def forward(self, img):
        # returns (batch_size, patch_num, dim), (batch_size, dim)
        batch_size = img.shape[0]
        res_fea = self.res_features(img)
        res_fea = rearrange(res_fea, "b d n1 n2 -> b (n1 n2) d")
        h = rearrange(res_fea, "b n d -> (b n) d")
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x, "(b n) d -> b n d", b=batch_size)
        out_pool = torch.mean(out_emb, dim=1)
        return out_emb, out_pool


class KADTextEncoder(nn.Module):
    """BERT + MLP text encoder matching KAD's CLP_clinical checkpoint layout."""

    def __init__(self, bert_model_name, embed_dim=768):
        super(KADTextEncoder, self).__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_embed_text(self, text):
        output = self.bert_model(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        last_hidden_state, pooler_output = output[0], output[1]
        emb_g = self.mlp_embed(pooler_output)
        emb_l = self.mlp_embed(last_hidden_state)
        return emb_l, emb_g


class BiMCQKAD(nn.Module):
    """KAD backbone (ResNet image encoder + clinical-BERT text encoder) wired into
    the same i2t/t2i MCQ fusion modules used by the CARZero/BiMCQ and MedKLIP
    backbones, so it can train against the same NIHBiMCQDataModule/build_t2i_mcq_batch
    pipeline."""

    def __init__(self, cfg):
        super(BiMCQKAD, self).__init__()

        self.cfg = cfg

        self.image_encoder = KADImageEncoder(res_base_model="resnet50")
        image_checkpoint = torch.load(cfg.model.KAD.image_ckpt_path, map_location="cpu")
        self.image_encoder.load_state_dict(image_checkpoint["image_encoder"])

        self.text_encoder = KADTextEncoder(bert_model_name=cfg.model.text.bert_type)
        text_checkpoint = torch.load(cfg.model.KAD.text_ckpt_path, map_location="cpu")
        self.text_encoder.load_state_dict(text_checkpoint["state_dict"], strict=False)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.t2i_fusion_module = builder.build_mcq_fusion_module(cfg, cfg.model.fusion.t2i_average_attn_weights)
        self.i2t_fusion_module = builder.build_mcq_fusion_module(cfg, cfg.model.fusion.i2t_average_attn_weights)

    def i2t_mcq_forward(self, x, i2t_only=False):
        img_emb_l, img_emb_g = self.image_encoder(x["imgs"])

        B, N, L = x["caption_ids"].shape
        caption_ids = x["caption_ids"].view(B * N, L)
        attention_mask = x["attention_mask"].view(B * N, L)

        text_emb_l, text_emb_g = self.text_encoder.encode_embed_text(
            {"input_ids": caption_ids, "attention_mask": attention_mask}
        )

        D = text_emb_g.shape[-1]
        text_emb_l = text_emb_l.view(B, N, text_emb_l.size(1), text_emb_l.size(2))
        text_emb_g = text_emb_g.view(B, N, text_emb_g.size(1))

        img_emb_ = torch.cat([img_emb_g.unsqueeze(1), img_emb_l], dim=1)
        text_emb_ = torch.cat([text_emb_g.unsqueeze(2), text_emb_l], dim=2)

        img_emb_g_ = img_emb_g.unsqueeze(1)
        img_emb_g_ = img_emb_g_.unsqueeze(1).expand(B, N, 1, D)
        img_emb_g_ = img_emb_g_.reshape(B * N, 1, D)
        img_emb_g_ = img_emb_g_.permute(1, 0, 2)  # (1, B*N, D)

        text_emb_g_ = text_emb_g.reshape(B * N, 1, D)
        text_emb_g_ = text_emb_g_.permute(1, 0, 2)  # (1, B*N, D)

        B_, S_img, D_ = img_emb_.shape
        _, N_, S_txt, _ = text_emb_.shape

        img_emb_ = img_emb_.permute(1, 0, 2)  # (S_img+1, B, D)
        img_emb_ = img_emb_.unsqueeze(2).expand(S_img, B, N, D)
        img_emb_ = img_emb_.reshape(S_img, B * N, D)
        img_emb_ = img_emb_.transpose(0, 1)  # (B*N, S_img, D)

        text_emb_ = text_emb_.permute(2, 0, 1, 3)  # (S_txt, B, N, D)
        text_emb_ = text_emb_.reshape(S_txt, B * N, D)
        text_emb_ = text_emb_.transpose(0, 1)  # (B*N, S_txt, D)

        if i2t_only:
            t2i_logit = None
        else:
            t2i_logit = self.t2i_fusion_module(img_emb_, text_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
            t2i_logit = t2i_logit.view(B, N)

        i2t_logit = self.i2t_fusion_module(text_emb_, img_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
        i2t_logit = i2t_logit.view(B, N)
        return i2t_logit, t2i_logit

    def t2i_mcq_forward(self, x, t2i_only=False):
        imgs = x["imgs"]
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N, C, H, W)

        img_emb_l, img_emb_g = self.image_encoder(imgs_flat)

        B_t, N_t, L = x["caption_ids"].shape
        assert B_t == B and N_t == N

        caption_ids = x["caption_ids"].reshape(B * N, L)
        attention_mask = x["attention_mask"].reshape(B * N, L)

        text_emb_l, text_emb_g = self.text_encoder.encode_embed_text(
            {"input_ids": caption_ids, "attention_mask": attention_mask}
        )

        _, S_img, D_img = img_emb_l.shape
        img_emb_l_bn = img_emb_l.view(B, N, S_img, D_img)
        img_emb_g_bn = img_emb_g.view(B, N, D_img)

        _, L_txt, D_txt = text_emb_l.shape
        text_emb_l_bn = text_emb_l.view(B, N, L_txt, D_txt)
        text_emb_g_bn = text_emb_g.view(B, N, D_txt)

        img_emb_l_flat = img_emb_l_bn.permute(0, 1, 3, 2)
        img_emb_l_flat = img_emb_l_flat.view(B * N, D_img, -1)
        img_emb_l_flat = img_emb_l_flat.permute(0, 2, 1)
        img_emb_g_pair = img_emb_g_bn.view(B * N, D_img)

        img_emb_ = torch.cat([img_emb_g_pair.unsqueeze(1), img_emb_l_flat], dim=1)  # (B*N, 1+S_img, D)

        text_emb_l_flat = text_emb_l_bn.view(B * N, L_txt, D_txt)
        text_emb_g_pair = text_emb_g_bn.view(B * N, D_txt)
        text_emb_ = torch.cat([text_emb_g_pair.unsqueeze(1), text_emb_l_flat], dim=1)  # (B*N, 1+L, D)

        img_emb_g_ = img_emb_g_pair.view(B * N, 1, D_img).permute(1, 0, 2)
        text_emb_g_ = text_emb_g_pair.view(B * N, 1, D_txt).permute(1, 0, 2)

        t2i_logit = self.t2i_fusion_module(img_emb_, text_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
        t2i_logit = t2i_logit.view(B, N)

        if t2i_only:
            i2t_logit = None
        else:
            i2t_logit = self.i2t_fusion_module(text_emb_, img_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1)
            i2t_logit = i2t_logit.view(B, N)

        return i2t_logit, t2i_logit

    def inference(self, images, text, mode="both"):
        B = images.shape[0]
        img_emb_l, img_emb_g = self.image_encoder(images)

        text_emb_l, text_emb_g = self.text_encoder.encode_embed_text(
            {"input_ids": text["caption_ids"], "attention_mask": text["attention_mask"]}
        )

        cls_bs = []
        for i in range(B):
            B_, L, D = text_emb_l.shape
            label_img_emb_l = img_emb_l[i:i + 1]
            label_img_emb_g = img_emb_g[i:i + 1].unsqueeze(1)

            query_emb_g_ = text_emb_g.unsqueeze(1).permute(1, 0, 2)

            attention_mask = text["attention_mask"].bool()
            global_valid = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([global_valid, attention_mask], dim=1)
            attention_mask = torch.logical_not(attention_mask)

            label_emb_ = torch.cat([label_img_emb_g, label_img_emb_l], dim=1)
            label_emb_ = label_emb_.expand(B_, -1, -1)
            query_emb_ = torch.cat([text_emb_g.unsqueeze(1), text_emb_l], dim=1)

            t2i_cls, t2i_attn, t2i_feats = self.t2i_fusion_module(
                label_emb_, query_emb_g_, inside_repeat=False, return_feat=True
            )
            t2i_cls = t2i_cls.squeeze(-1).squeeze(-1)

            label_emb_g__ = label_img_emb_g.expand(-1, B_, -1)
            i2t_cls, i2t_attn, i2t_feats = self.i2t_fusion_module(
                query_emb_, label_emb_g__, inside_repeat=False, attention_mask=attention_mask, return_feat=True
            )
            i2t_cls = i2t_cls.squeeze(-1).squeeze(-1)

            if mode == "i2t":
                cls = i2t_cls
            elif mode == "t2i":
                cls = t2i_cls
            else:
                cls = (i2t_cls + t2i_cls) / 2

            cls_bs.append(cls.unsqueeze(0).detach().cpu())

        cls = torch.cat(cls_bs, dim=0)
        return cls.numpy()

    def process_text(self, text, device):
        if isinstance(text, str):
            text = [text]

        processed_text_tensors = []
        for t in text:
            t = t.replace("\n", " ")
            splitter = re.compile(r"[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []
            for t_sent in captions:
                t_sent = t_sent.replace("��", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t_sent.lower())
                if len(tokens) <= 1:
                    continue
                included_tokens = []
                for tok in tokens:
                    tok = tok.encode("ascii", "ignore").decode("ascii")
                    if len(tok) > 0:
                        included_tokens.append(tok)
                all_sents.append(" ".join(included_tokens))

            t_joined = " ".join(all_sents)
            text_tensors = self.tokenizer(
                t_joined,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack([x["attention_mask"] for x in processed_text_tensors])
        token_type_ids = torch.stack([x["token_type_ids"] for x in processed_text_tensors])

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = [len([w for w in txt if not w.startswith("[")]) for txt in text]

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):
        return {k: self.process_text(v, device) for k, v in class_prompts.items()}


class BiMCQKADModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.KAD_model = None
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=15, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']

        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        self.prompts = [*self.pos_prompts.values(), *self.neg_prompts.values()]
        self.prompts = [f"There is {cls.replace('_', ' ')} but no {neg_cls.replace('_', ' ')}." for cls in self.class_names[:-1] for neg_cls in self.class_names[:-1] if cls != neg_cls] + self.prompts

    def setup(self, stage=None):
        if self.KAD_model is None:
            self.KAD_model = BiMCQKAD(self.cfg)
            self.freeze_module()
            self.print("KAD model loaded and frozen.")
        if self.dm is None:
            self.dm = self.trainer.datamodule

    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.KAD_model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.KAD_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            for param in self.KAD_model.i2t_fusion_module.parameters():
                param.requires_grad = False
            for param in self.KAD_model.t2i_fusion_module.parameters():
                param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.KAD_model.image_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.KAD_model.text_encoder.parameters()))
        self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.KAD_model.i2t_fusion_module.parameters()))
        self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.KAD_model.t2i_fusion_module.parameters()))

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.KAD_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def i2t_forward(self, batch):
        i2t_cls, t2i_cls = self.KAD_model.i2t_mcq_forward(batch, i2t_only=self.cfg.model.KAD.single_path)

        logits = (i2t_cls + t2i_cls) / 2 if self.cfg.model.KAD.single_path == False else i2t_cls

        targets = batch["answer_idx"].to(self.device)

        loss = F.cross_entropy(logits, targets, reduction="mean")
        acc = (logits.argmax(dim=1) == targets).float().mean()

        return i2t_cls, t2i_cls, loss, acc

    def t2i_forward(self, batch):
        batch = build_t2i_mcq_batch(
            batch,
            self.tokenizer,
            self.prompts,
            self.class_names,
            max_length=self.cfg.data.text.word_num,
            num_negatives=2,
            no_hyb=self.cfg.data.text.no_hyb
            )

        if len(batch['imgs'].shape) != 5 :
            self.print(f"Unexpected image batch shape: {batch['imgs'].shape}")
            return None, None, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        i2t_cls, t2i_cls = self.KAD_model.t2i_mcq_forward(batch, t2i_only=self.cfg.model.KAD.single_path)

        logits = (i2t_cls + t2i_cls) / 2 if self.cfg.model.KAD.single_path == False else t2i_cls

        acc = (logits.argmax(dim=1) == batch["answer_idx"].to(self.device)).float().mean()

        loss = F.cross_entropy(logits, batch["answer_idx"].to(self.device), reduction="mean")

        return i2t_cls, t2i_cls, loss, acc

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bce_loss = self.metrics(batch, "val")
        return {"val/bce_loss": bce_loss.detach()}

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        bce_loss = self.metrics(batch, "test")
        return {"test/loss": loss.detach(), "test/bce_loss": bce_loss.detach()}

    def shared_step(self, batch, split):
        weight = self.cfg.train.loss_weight

        i2t_logits_i2t, t2i_logits_i2t, i2t_loss, i2t_acc = self.i2t_forward(batch)
        i2t_logits_t2i, t2i_logits_t2i, t2i_loss, t2i_acc = self.t2i_forward(batch)

        ce_loss = weight * i2t_loss + (1 - weight) * t2i_loss

        self.log_dict({f"{split}/loss": ce_loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc},
                  prog_bar=True, on_epoch=True, sync_dist=True)

        return ce_loss

    def metrics(self, batch, split):
        imgs = batch["imgs"].to(self.device)
        labels = batch["label"].to(self.device)

        pos_text = self.KAD_model.process_text(self.dm.train_dataset.pos_prompts, self.device)
        pos_logits = self.KAD_model.inference(imgs, pos_text)
        pos_logits = torch.tensor(pos_logits, device=self.device)
        pos_probs = torch.sigmoid(pos_logits)

        neg_text = self.KAD_model.process_text(self.dm.train_dataset.neg_prompts, self.device)
        neg_logits = self.KAD_model.inference(imgs, neg_text)  # (N, 14)
        neg_logits = torch.tensor(neg_logits, device=self.device)
        neg_probs = torch.sigmoid(neg_logits)
        neg_targets = (1 - labels[:, :-1]).int()

        self.auroc_metric.update(pos_probs, labels.int())
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, labels)
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())
        weight = self.cfg.train.loss_weight
        bce_loss = weight * pos_bce_loss + (1 - weight) * neg_bce_loss

        class_auroc = self.auroc_metric.compute()
        neg_class_auroc = self.neg_auroc_metric.compute()
        pos_mean_auroc = class_auroc.mean()
        neg_mean_auroc = neg_class_auroc.mean()
        mean_auroc = (pos_mean_auroc + neg_mean_auroc) / 2

        self.log(f"{split}/bce_loss", bce_loss, prog_bar=True, sync_dist=True)
        self.log(f"{split}/mean_auroc", mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/pos_mean_auroc", pos_mean_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{split}/neg_mean_auroc", neg_mean_auroc, prog_bar=True, sync_dist=True)

        self.log_dict({f"{split}/auroc_{c}": class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names)}, sync_dist=True)
        self.log_dict({f"{split}/neg_auroc_{c}": neg_class_auroc[i]
                       for i, c in enumerate(self.dm.train_dataset.class_names[:-1])}, sync_dist=True)

        return bce_loss

    def on_before_optimizer_step(self, optimizer):
        total_norm_sq = sum(
            p.grad.detach().data.norm(2).item() ** 2
            for p in self.KAD_model.parameters() if p.grad is not None
        )
        self.log("grad_norm", total_norm_sq ** 0.5, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.trainer.is_global_zero:
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            for key in ["val/loss", "val/bce_loss", "val/mean_auroc", "val/pos_mean_auroc", "val/neg_mean_auroc"]:
                if key in metrics:
                    self.print(f" - {key:<17}: {metrics[key].item():.4f}")

            self.print(" - Class-wise AUROC:")
            class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/auroc_")}
            for key in class_metrics:
                class_name = key.replace("val/auroc_", "")
                self.print(f"    {class_name:<22}: {class_metrics[key].item():.4f}")

            self.print(" - Negative Class-wise AUROC:")
            neg_class_metrics = {k: v for k, v in metrics.items() if k.startswith("val/neg_auroc_")}
            for key in neg_class_metrics:
                class_name = key.replace("val/neg_auroc_", "")
                self.print(f"    {class_name:<22}: {neg_class_metrics[key].item():.4f}")

    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.dm.train_dataset.class_names):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.dm.train_dataset.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
