import os
import torch
import numpy as np
import copy
import random
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn.functional  as F
from . import builder
from . import utils
from . import constants
from .models.vision_model import PretrainedImageClassifier
from typing import Union, List
from scipy import ndimage
from tqdm import tqdm
import time
import yaml
from easydict import EasyDict
import matplotlib.pyplot as plt

np.random.seed(10)
random.seed(10)

_MODELS = {
    "CARZero_resnet50": "",
    "CARZero_vit_b_16": "pretrain_model/CARZero_best_model.ckpt",
}

_FEATURE_DIM = {"CARZero_resnet50": 2048, "CARZero_vit_b_16": 768 }

def available_models() -> List[str]:
    """Returns the names of available CARZero models"""
    return list(_MODELS.keys())


def load_CARZero(
    name: str = "CARZero_vit_b_16",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    multi: bool = False,
    cfg: yaml = None,
):
    """Load a CARZero model

    Parameters
    ----------
    name : str
        A model name listed by `CARZero.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    CARZero_model : torch.nn.Module
        The CARZero model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Make sure to download the pretrained weights from \n"
            + "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            + " and copy it to the ./pretrained folder."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    #cfg = ckpt["hyper_parameters"]
    if cfg is None:
        with open(os.path.join(os.path.dirname(ckpt_path), 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)
    else :
        cfg = EasyDict(cfg)
    ckpt_dict = ckpt["state_dict"]

    # CARZero_model = builder.build_CARZero_dqn_llm_model(cfg).to(device)
    # CARZero_model = builder.build_CARZero_dqn_wo_self_atten_model(cfg).to(device)
    if not multi:
        CARZero_model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(cfg).to(device)
    else :
        CARZero_model = builder.build_BiMCQ_model(cfg).to(device)
    # CARZero_model = builder.build_CARZero_dqn_self_atten_local_model(cfg).to(device)

    model_weights = CARZero_model.state_dict()

    # 키 이름 정리 및 일치하는 가중치만 선택
    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("CARZero.")[-1]
        if new_key in model_weights and model_weights[new_key].shape == v.shape:
            fixed_ckpt_dict[new_key] = v

    # 선택된 가중치만 로드
    CARZero_model.load_state_dict(fixed_ckpt_dict, strict=False)

    print(f"Loaded {len(fixed_ckpt_dict)}/{len(model_weights)} layers successfully.")
    return CARZero_model

def load_BiMCQ(
    name: str = "CARZero_vit_b_16",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    multi: bool = False,
    cfg: yaml = None,
):
    """Load a BiMCQ model

    Parameters
    ----------
    name : str
        A model name listed by `CARZero.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    BiMCQ_model : torch.nn.Module
        The BiMCQ model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Make sure to download the pretrained weights from \n"
            + "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            + " and copy it to the ./pretrained folder."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    #cfg = ckpt["hyper_parameters"]
    if cfg is None:
        with open(os.path.join(os.path.dirname(ckpt_path), 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)
    else :
        cfg = EasyDict(cfg)
    ckpt_dict = ckpt["state_dict"]

    BiMCQ_model = builder.build_BiMCQ_model(cfg).to(device)

    model_weights = BiMCQ_model.state_dict()

    # 키 이름 정리 및 일치하는 가중치만 선택
    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("CARZero.")[-1]
        if new_key in model_weights and model_weights[new_key].shape == v.shape:
            fixed_ckpt_dict[new_key] = v

    # 선택된 가중치만 로드
    BiMCQ_model.load_state_dict(fixed_ckpt_dict, strict=False)

    print(f"Loaded {len(fixed_ckpt_dict)}/{len(model_weights)} layers successfully.")
    return BiMCQ_model

def load_img_classification_model(
    name: str = "CARZero_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    num_cls: int = 1,
    freeze_encoder: bool = True,
):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `CARZero.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    num_cls: int
        Number of output classes
    freeze_encoder: bool
        Freeze the pretrained image encoder

    Returns
    -------
    img_model : torch.nn.Module
        The CARZero pretrained image classification model
    """

    # load pretrained image encoder
    CARZero_model = load_CARZero(name, device)
    image_encoder = copy.deepcopy(CARZero_model.img_encoder)
    del CARZero_model

    # create image classifier
    feature_dim = _FEATURE_DIM[name]
    img_model = PretrainedImageClassifier(
        image_encoder, num_cls, feature_dim, freeze_encoder
    )

    return img_model

def get_similarities(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both", 'atten']:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g = CARZero_model.image_encoder_forward(imgs)
        text_emb_l, text_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

    # get similarities
    global_similarities = CARZero_model.get_global_similarities(img_emb_g, text_emb_g)
    # ipdb.set_trace()
    local_similarities, attention_maps = CARZero_model.get_local_similarities(
        img_emb_l, text_emb_l, txts["cap_lens"], return_atten=True
    )
    similarities = (local_similarities + global_similarities) / 2

    # ipdb.set_trace()

    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    elif similarity_type == "both":
        return similarities.detach().cpu().numpy()
    elif similarity_type == 'atten':
        attention_maps = torch.from_numpy(attention_maps.repeat(16, axis=1).repeat(16, axis=2))#Final 
        return attention_maps

def get_dqn_similarities(CARZero_model, imgs, txts, similarity_type="both", dropout=False, atten_map=False, feature=False, mode=None, multi=False):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        CARZero_model.eval() if not dropout else CARZero_model.train()
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

        cls_bs = []
        i2t_SimR_bs = []
        t2i_SimR_bs = []
        i2t_atten_bs = []
        t2i_atten_bs = []
        bs = label_img_emb_g.size(0)
        for i in range(bs):
            label_img_emb_l_ = label_img_emb_l[i:i+1].view(label_img_emb_l[i:i+1].size(0), label_img_emb_l[i:i+1].size(1), -1)  # (1, D, S_img)

            label_img_emb_g_ = label_img_emb_g[i:i+1] # (1, D)

            label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) # (1, S_img, D)

            query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) # (B, D, S_txt)

            query_emb_l_ = query_emb_l_.permute(0, 2, 1) # (B, S_txt, D)

            attention_mask = txts['attention_mask'].bool() # (B, S_txt)
            global_valid = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([global_valid, attention_mask], dim=1)
            attention_mask = torch.logical_not(attention_mask)

            if not multi:
                t2i_cls, atten_t2i, t2i_feat = CARZero_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_feat=True)
            else :
                t2i_cls, atten_t2i, t2i_feat = CARZero_model.t2i_fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_feat=True)

            t2i_cls = t2i_cls.squeeze(-1)

            if not multi:
                i2t_cls, atten_i2t, i2t_feat = CARZero_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_feat=True, attention_mask=attention_mask)
            else :
                i2t_cls, atten_i2t, i2t_feat = CARZero_model.i2t_fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_feat=True, attention_mask=attention_mask)

            i2t_cls = i2t_cls.squeeze(-1).transpose(1,0) 

            i2t_atten_bs.append(torch.stack(atten_i2t).mean(dim=0))
            t2i_atten_bs.append(torch.stack(atten_t2i).mean(dim=0))
            
            if mode == 'i2t' :
                cls = i2t_cls
            elif mode == 't2i':
                cls = t2i_cls
            else:
            # cls = t2i_g_cls
                cls = (i2t_cls + t2i_cls) / 2

            cls_bs.append(cls)
            i2t_SimR_bs.append(i2t_feat.squeeze())
            t2i_SimR_bs.append(t2i_feat.squeeze())
        cls = torch.cat(cls_bs, dim=0)
        
        if atten_map:
            atten_i2t = torch.stack(i2t_atten_bs)
            atten_t2i = torch.stack(t2i_atten_bs)
            return cls.detach().cpu().numpy(), atten_i2t.detach().cpu().numpy(), atten_t2i.detach().cpu().numpy()
        
        if feature:
            i2t_SimR_bs = torch.stack(i2t_SimR_bs)
            t2i_SimR_bs = torch.stack(t2i_SimR_bs)
            return cls.detach().cpu().numpy(), i2t_SimR_bs.detach().cpu().numpy(), t2i_SimR_bs.detach().cpu().numpy()

        return cls.detach().cpu().numpy()

def bimcq_classification(CARZero_model, imgs, txts, similarity_type="both", dropout=False, feature=False, atten_map=False, mode='both'):
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    CARZero_model.eval() if not dropout else CARZero_model.train()
    label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
    query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
        txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
    )

    cls_bs = []
    i2t_cls_bs = []
    t2i_cls_bs = []
    i2t_attns_bs = []
    t2i_attns_bs = []
    i2t_feats_bs = []
    t2i_feats_bs = []
    
    bs = label_img_emb_g.size(0)
    for i in range(bs):
        B, S_txt, D = query_emb_l.size()
        label_img_emb_l_ = label_img_emb_l[i:i+1].view(label_img_emb_l[i:i+1].size(0), label_img_emb_l[i:i+1].size(1), -1)  # (1, D, S_img)

        label_img_emb_g_ = label_img_emb_g[i:i+1] # (1, D)

        label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) # (1, S_img, D)

        query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) # (B, D, S_txt)

        query_emb_l_ = query_emb_l_.permute(0, 2, 1) # (B, S_txt, D)
        
        query_emb_g_ = query_emb_g.unsqueeze(1).permute(1,0,2)  # (1, B, D)
        label_img_emb_g_ = label_img_emb_g_.unsqueeze(1) # (1, 1, D)

        attention_mask = txts['attention_mask'].bool() # (B, S_txt)
        global_valid = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype, device=attention_mask.device) # (B, 1)
        attention_mask = torch.cat([global_valid, attention_mask], dim=1) # (B, S_txt+1)
        attention_mask = torch.logical_not(attention_mask) # (B, S_txt+1)

        label_emb_ = torch.cat([label_img_emb_g_, label_img_emb_l_], dim=1) # (1, 1+S_img, D)
        label_emb_ = label_emb_.expand(B, -1, -1)
        query_emb_ = torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1) # (B, S_txt+1, D)
        t2i_cls, t2i_attn, t2i_feats = CARZero_model.t2i_fusion_module(label_emb_, query_emb_g_, inside_repeat=False, return_feat=True)
        # (B, 1+S_img, D) * (1, B, D) -> (B, 1, 1)
        
        t2i_cls = t2i_cls.squeeze(-1).squeeze(-1)  # (B,)

        label_emb_g__ = label_img_emb_g_.expand(-1, B, -1) # (1, B, D)
        i2t_cls, i2t_attn, i2t_feats = CARZero_model.i2t_fusion_module(query_emb_, label_emb_g__, inside_repeat=False, attention_mask=attention_mask, return_feat=True)
        # (B, S_txt+1, D) * (1, B, D) -> (B, 1, 1)

        # ipdb.set_trace()
        i2t_cls = i2t_cls.squeeze(-1).squeeze(-1)  # (B,)
        
        i2t_attns_bs.append(torch.stack(i2t_attn).mean(dim=0).detach().cpu())
        t2i_attns_bs.append(torch.stack(t2i_attn).mean(dim=0).detach().cpu())
        
        i2t_feats_bs.append(i2t_feats.squeeze().detach().cpu())
        t2i_feats_bs.append(t2i_feats.squeeze().detach().cpu())

        if mode == 'i2t' :
            cls = i2t_cls
        elif mode == 't2i':
            cls = t2i_cls
        else:
            cls = (i2t_cls + t2i_cls) / 2
        cls_bs.append(cls.detach().cpu().unsqueeze(0))
        i2t_cls_bs.append(i2t_cls.detach().cpu().unsqueeze(0))
        t2i_cls_bs.append(t2i_cls.detach().cpu().unsqueeze(0))
    
    cls = torch.cat(cls_bs, dim=0) # (N, B)
    if atten_map:
        atten_i2t = torch.stack(i2t_attns_bs)
        atten_t2i = torch.stack(t2i_attns_bs)
        return cls.detach().cpu().numpy(), atten_i2t.numpy(), atten_t2i.numpy()
    
    if feature:
        i2t_feats = torch.stack(i2t_feats_bs)
        t2i_feats = torch.stack(t2i_feats_bs)
        return cls.detach().cpu().numpy(), i2t_feats.numpy(), t2i_feats.numpy()
    
    return cls.detach().cpu().numpy()

def dqn_shot_classification(CARZero_model, imgs, cls_txt_mapping, dropout=False, feature=False, atten_map=False, mode=None, multi=False, mcq=False, torch_tensor=False, seperate=False, ts=False):
    """Load a CARZero pretrained classification model
    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """
    text_batch = cls_txt_mapping
    
    if feature :
        return get_dqn_similarities(
            CARZero_model, imgs, text_batch, similarity_type="both", dropout=dropout, feature=feature, mode=mode, multi=multi
        )
    elif atten_map:
        return get_dqn_similarities(
            CARZero_model, imgs, text_batch, similarity_type="both", dropout=dropout, atten_map=atten_map, mode=mode, multi=multi
        )
    else:
        return get_dqn_similarities(
            CARZero_model, imgs, text_batch, similarity_type="both", dropout=dropout, feature=feature, mode=mode, multi=multi
        )