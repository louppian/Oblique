"""CARZero/BiMCQ backbone adapter: model loading + classification calls plugged into
the shared inference/common.py pipeline."""
import torch
from omegaconf import OmegaConf

import CARZero


def load_model(args):
    cfg = OmegaConf.load(args.cfg_path)

    if args.ckpt_path:
        model = CARZero.load_CARZero(
            name="CARZero_vit_b_16", device=args.device, multi=cfg.model.CARZero.multi, cfg=cfg
        )
        ckpt_state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        fixed_ckpt_dict = {
            k.split("CARZero_model.")[-1]: v
            for k, v in ckpt_state_dict.items()
            if k.split("CARZero_model.")[-1] in model.state_dict()
        }
        model.load_state_dict(fixed_ckpt_dict, strict=True)
    else:
        model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=args.device, multi=False, cfg=cfg)

    model.eval()
    return model, cfg


def classify(model, imgs, txt):
    return CARZero.bimcq_classification(model, imgs, txt)


def classify_feature(model, imgs, txt, mode):
    return CARZero.bimcq_classification(model, imgs, txt, feature=True, mode=mode)
