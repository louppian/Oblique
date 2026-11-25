"""MedKLIP backbone adapter: model loading + classification calls plugged into the
shared inference/common.py pipeline."""
import torch
from omegaconf import OmegaConf

from finetune.models import BiMCQMedKLIP


def load_model(args):
    cfg = OmegaConf.load(args.cfg_path)
    model = BiMCQMedKLIP(cfg)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = {k.replace("MedKLIP_model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()
    return model, cfg


def classify(model, imgs, txt):
    return model.inference(imgs, txt)
