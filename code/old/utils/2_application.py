#!/usr/bin/env python3
"""섹션 2 — 응용과 점수 해석: 실험 함수 3개.
exp07: 긍정 점수 M/V/A/U 분해, 분리도–AUC 관계, 성분 제거.
exp08: 10쌍 부정 템플릿의 이동, 점수 차이 분해, 질환별 AUC.
exp09: 공격 탐지와 예산별 임베딩 이동, 여러 노이즈 꼭지점 비교.
AppConfig를 수정하고 python 2_application.py로 실행한다. 1_analysis.py와 같은 폴더.
결과: <root>/results/section_experiments/application/ — 실험별 CSV/PNG 하나.
분류기 예측을 공격 라벨로 사용한다. CW는 L2 공격으로 Linf 예산의 공격과 구분한다.
공격 성공 여부와 실제 Linf 변화량을 별도 기록한다.
Dependencies: 1_analysis.py의 패키지 + torch torchvision timm torchattacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code').is_dir() and (p / 'third_party').is_dir())

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt

import importlib
an = importlib.import_module("1_analysis")
Axes = an.Axes
normalize = an.normalize
auroc = an.auroc
bootstrap_mean_ci = an.bootstrap_mean_ci
save_csv = an.save_csv
save_figure = an.save_figure
legend_below = an.legend_below
band = an.band
NIH_CLASSES = an.NIH_CLASSES
C1 = an.C1
C2 = an.C2
C3 = an.C3
C4 = an.C4
C5 = an.C5
MUTED = an.MUTED
INK = an.INK


# =============================================================================
# 설정
# =============================================================================
A4D_PROMPTS = (
    "An image with noise", "An image with unnatural texture", "An image with digital distortions",
    "An image with corrupted pixels", "An image with adversarial perturbations",
    "An image that was modified artificially", "An image with strange visual artifacts",
    "An image with high-frequency noise", "An image that contains tampering", "An image with pixel-level manipulation",
)


@dataclass(frozen=True)
class AppConfig(an.Config):
    tiny_root: Path = ROOT / 'data' / 'tiny-imagenet-200'
    datasets: tuple[str, ...] = ("coco", "tiny")
    classifiers: tuple[str, ...] = ("wide_resnet50_2", "resnet34", "deit_small", "convnext_tiny")
    # DeepFool 제외: torchattacks 구현은 이미지 1장씩, 단계마다 1,000개 클래스 기울기를 역전파로 구해
    # ImageNet 분류기에서 실행 1회에 수 시간이 걸린다. 또한 ε 예산이 없는 L2 공격이다.
    attacks: tuple[str, ...] = ("FGSM", "PGD", "BIM", "AutoAttack", "CW", "Square")
    eps: float = 8 / 255
    pgd_eps_sweep: tuple[float, ...] = (1 / 255, 2 / 255, 4 / 255, 8 / 255)
    n_attack_images: int = 500          # 데이터셋마다 공격할 테스트 이미지 수
    n_reference_clean: int = 200        # A⁴D 표준화 / r̂ 추정용 깨끗한 기준 이미지 (테스트와 분리)
    attack_batch: int = 128
    run_p1: bool = True
    run_p2: bool = True


APP = AppConfig()
ATTACK_COLOR = dict(FGSM=C1, PGD=C2, BIM=C3, AutoAttack=C4, DeepFool=C5, CW="#008300", Square="#4a3aa7")
SCORE_COLOR = {"n": C3, "−h": C2, "A4D": MUTED}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# =============================================================================
# P1. 적대적 공격 탐지
# =============================================================================
def load_classifier(name: str, device: str):
    """ImageNet 사전학습 분류기 + 정규화 래퍼 (입력: [0,1] 픽셀). 검증 시 교체 가능한 주입 지점."""
    import torch
    import torch.nn as nn
    if name == "deit_small":
        import timm
        net = timm.create_model("deit_small_patch16_224", pretrained=True)
    else:
        import torchvision
        net = getattr(torchvision.models, name)(weights="DEFAULT")

    class Normalized(nn.Module):
        def __init__(self, net):
            super().__init__(); self.net = net
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x):
            return self.net((x - self.mean) / self.std)
    return Normalized(net).to(device).eval()


CLASSIFIER_LOADER = load_classifier


def attack_pixels(path, size: int = 224) -> np.ndarray:
    """짧은 변 resize → 중앙 crop 224 → [0,1]. 분류기·CLIP이 같은 픽셀을 본다 (Tiny 64px는 224로 확대)."""
    with Image.open(path) as im:
        im = im.convert("RGB"); w, h = im.size; s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size; left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
    return np.asarray(im, dtype=np.float32) / 255


def dataset_paths(cfg: AppConfig, ctx: an.Context, dataset: str) -> list[Path]:
    if dataset == "coco":
        return [ctx.data.coco_path(i) for i in sorted(ctx.data.coco_images)]
    ann = cfg.tiny_root / "val" / "val_annotations.txt"
    names = [line.split("\t")[0] for line in ann.read_text().splitlines() if line.strip()]
    return [cfg.tiny_root / "val" / "images" / n for n in sorted(names)]


def make_attack(name: str, model, eps: float, n_classes: int = 1000):
    import torchattacks as ta
    if name == "FGSM":
        return ta.FGSM(model, eps=eps)
    if name == "PGD":
        return ta.PGD(model, eps=eps, alpha=eps / 4, steps=10, random_start=True)
    if name == "BIM":
        return ta.BIM(model, eps=eps, alpha=eps / 4, steps=10)
    if name == "AutoAttack":
        return ta.AutoAttack(model, norm="Linf", eps=eps, version="standard", n_classes=n_classes, seed=0)
    if name == "DeepFool":
        return ta.DeepFool(model, steps=50, overshoot=0.02)
    if name == "CW":
        return ta.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    if name == "Square":
        return ta.Square(model, norm="Linf", eps=eps, n_queries=5000, seed=0)
    raise ValueError(name)


def run_attack(ctx: an.Context, cfg: AppConfig, dataset: str, clf_name: str, attack: str, eps: float,
               paths: list[Path], key: str) -> dict:
    """공격 이미지의 CLIP 임베딩 + 성공 여부 + L∞ (이미지는 저장하지 않는다)."""
    import torch
    clip = ctx.encoder("clip")
    meta = dict(kind="attack", dataset=dataset, classifier=clf_name, attack=attack, eps=eps, key=key,
                files=[str(p.name) for p in paths], clip=clip.signature, torchattacks=an.importlib.metadata.version("torchattacks"))
    side = ctx.cache / f"attack_{dataset}_{clf_name}_{attack}_{eps:.5f}_{key}_meta.npz"
    holder = {}

    def compute():
        model = CLASSIFIER_LOADER(clf_name, clip.device)
        with torch.no_grad():
            n_classes = int(model(torch.zeros(1, 3, 224, 224, device=clip.device)).shape[1])
        atk = make_attack(attack, model, eps, n_classes)
        embs, success, linf, labels = [], [], [], []
        for i in range(0, len(paths), cfg.attack_batch):
            x = torch.from_numpy(np.stack([attack_pixels(p) for p in paths[i:i + cfg.attack_batch]]).transpose(0, 3, 1, 2).copy()).to(clip.device)
            # contiguous 필수: torchattacks FAB(AutoAttack 내부)가 입력 stride를 그대로 쓰는 .view에서 실패한다.
            with torch.no_grad():
                y = model(x).argmax(1)
            xa = atk(x, y).detach()
            with torch.no_grad():
                ya = model(xa).argmax(1)
            success.append((ya != y).cpu().numpy()); labels.append(y.cpu().numpy())
            linf.append((xa - x).abs().flatten(1).max(1).values.cpu().numpy())
            embs.append(clip.encode_pixels(xa.cpu().numpy().transpose(0, 2, 3, 1)))
        holder.update(success=np.concatenate(success), linf=np.concatenate(linf), labels=np.concatenate(labels))
        an.save_npz(side, **holder)
        return np.concatenate(embs)
    raw = ctx.cached(f"attack_{dataset}_{clf_name}_{attack}_{key}", meta, compute)
    if not holder:
        with np.load(side) as z:
            holder = {k: z[k] for k in z.files}
    return dict(Z=normalize(raw), **holder)


def a4d_scorer(P_ref: np.ndarray):
    """A⁴D: 기준 이미지 유사도 행렬을 열마다 표준화 → PC1. PC1 부호는 적재값 합이 양수가 되도록 정한다
    (노이즈를 묘사한 프롬프트와 더 비슷할수록 공격 점수가 커지게)."""
    mu, sd = P_ref.mean(0), P_ref.std(0) + 1e-12
    Pt = (P_ref - mu) / sd
    _, _, Vt = np.linalg.svd(Pt - Pt.mean(0), full_matrices=False)
    v1 = Vt[0] * np.sign(Vt[0].sum() or 1.0)
    return lambda P: ((P - mu) / sd) @ v1


def _section_p1(ctx: an.Context, cfg: AppConfig) -> None:
    an._banner("P1", "Adversarial attack detection with the noise point (CLIP)")
    clip = ctx.encoder("clip")
    b = ctx.pooled_b("clip", "image")
    T = normalize(clip.encode_texts(list(A4D_PROMPTS)))
    classifiers, attacks, n_test = cfg.classifiers, cfg.attacks, cfg.n_attack_images
    rows, mech, sweep = [], [], []
    for dataset in cfg.datasets:
        paths_all = dataset_paths(cfg, ctx, dataset)
        perm = np.random.default_rng(4040).permutation(len(paths_all))
        ref_paths = [paths_all[i] for i in perm[:cfg.n_reference_clean]]
        test_paths = [paths_all[i] for i in perm[cfg.n_reference_clean:cfg.n_reference_clean + n_test]]
        meta = dict(kind="attack_clean", dataset=dataset, files=[p.name for p in ref_paths + test_paths], clip=clip.signature)
        Zall = normalize(ctx.cached(f"attack_clean_{dataset}", meta,
                                    lambda: clip.encode_pixels(attack_pixels(p) for p in ref_paths + test_paths)))
        Zref, Zclean = Zall[:len(ref_paths)], Zall[len(ref_paths):]
        A = Axes(Zref.mean(0), b)
        a4d = a4d_scorer(Zref @ T.T)
        def scores(Z):
            S = A.coords(Z)
            return {"n": S["n"], "−h": -S["h"], "A4D": a4d(Z @ T.T), "c": S["c"], "h": S["h"]}
        sc_clean = scores(Zclean)
        for clf_name in classifiers:
            for attack in attacks:
                res = run_attack(ctx, cfg, dataset, clf_name, attack, cfg.eps, test_paths, f"n{n_test}")
                sc_adv = scores(res["Z"]); rng = np.random.default_rng(1212)
                for subset in ("all", "successful"):
                    keep = np.ones(len(res["success"]), bool) if subset == "all" else res["success"].astype(bool)
                    for s in ("n", "−h", "A4D"):
                        pos, neg = sc_adv[s][keep], sc_clean[s]
                        point = auroc(pos, neg) if keep.sum() else np.nan
                        boots = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
                                 for _ in range(cfg.bootstrap_repeats)] if keep.sum() > 1 else [np.nan]
                        rows.append(dict(dataset=dataset, classifier=clf_name, attack=attack,
                                         eps=cfg.eps if attack not in ('CW','DeepFool') else np.nan,
                                         attack_norm='L2' if attack in ('CW','DeepFool') else 'Linf',
                                         eps_constrained=attack not in ('CW','DeepFool'),subset=subset,
                                         score=s, n_clean=len(neg), n_adv=int(keep.sum()), auroc=point,
                                         ci_low=float(np.nanquantile(boots, 0.025)), ci_high=float(np.nanquantile(boots, 0.975))))
                d = {k: sc_adv[k] - sc_clean[k] for k in ("c", "n", "h")}
                mech.append(dict(dataset=dataset, classifier=clf_name, attack=attack, attack_success_rate=float(res["success"].mean()),
                                 mean_linf_255=float(res["linf"].mean() * 255),
                                 **{f"delta_{k}_mean": float(v.mean()) for k, v in d.items()},
                                 **{f"delta_{k}_ci_low": bootstrap_mean_ci(v, cfg.bootstrap_repeats, rng)[0] for k, v in d.items()},
                                 **{f"delta_{k}_ci_high": bootstrap_mean_ci(v, cfg.bootstrap_repeats, rng)[1] for k, v in d.items()},
                                 clean_c_mean=float(sc_clean["c"].mean()), clean_n_mean=float(sc_clean["n"].mean()),
                                 adv_c_mean=float(sc_adv["c"].mean()), adv_n_mean=float(sc_adv["n"].mean())))
                ctx.summary.setdefault("_dh", {})[(dataset, clf_name, attack)] = d["h"]
                print(f"  {dataset} / {clf_name} / {attack}: success = {res['success'].mean():.3f}, "
                      f"AUROC n = {[r['auroc'] for r in rows if r['dataset'] == dataset and r['classifier'] == clf_name and r['attack'] == attack and r['subset'] == 'all' and r['score'] == 'n'][0]:.3f}")
        sweep_clf = "resnet34" if "resnet34" in classifiers else classifiers[0]
        for eps in cfg.pgd_eps_sweep:
            res = run_attack(ctx, cfg, dataset, sweep_clf, "PGD", eps, test_paths, f"n{n_test}")
            sc_adv = scores(res["Z"]); rng = np.random.default_rng(1313)
            for s in ("n", "−h", "A4D"):
                pos, neg = sc_adv[s], sc_clean[s]
                boots = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
                         for _ in range(cfg.bootstrap_repeats)]
                sweep.append(dict(dataset=dataset, classifier=sweep_clf, eps_255=eps * 255, score=s, auroc=auroc(pos, neg),
                                  ci_low=float(np.quantile(boots, 0.025)), ci_high=float(np.quantile(boots, 0.975)),
                                  attack_success_rate=float(res["success"].mean())))
    tab_s = pd.DataFrame(sweep)
    tab1 = save_csv(ctx.out, "tab1_attack_auroc",
                    pd.concat([pd.DataFrame(rows), tab_s.assign(attack="PGD_eps_sweep", subset="all")], ignore_index=True))
    tab2 = save_csv(ctx.out, "tab2_attack_mechanism", mech)
    ctx.summary["P1"] = {f"{r.dataset}/{r.classifier}/{r.attack}/{r.score}": r.auroc
                         for r in tab1[(tab1.subset == "all") & (tab1.attack != "PGD_eps_sweep")].itertuples()}
    if not cfg.make_figures:
        return
    fig, axes = plt.subplots(1, len(cfg.datasets), figsize=(7 * len(cfg.datasets), 4.6), squeeze=False)
    for ax, dataset in zip(axes[0], cfg.datasets):
        sub = tab1[(tab1.dataset == dataset) & (tab1.subset == "all") & (tab1.attack != "PGD_eps_sweep")]
        g = sub.groupby(["attack", "score"], sort=False).agg(auroc=("auroc", "mean"), lo=("ci_low", "mean"),
                                                            hi=("ci_high", "mean")).reset_index()
        att = list(dict.fromkeys(g.attack)); width = 0.26
        for k, s in enumerate(("n", "−h", "A4D")):
            gg = g[g.score == s].set_index("attack").loc[att]
            ax.bar(np.arange(len(att)) + (k - 1) * width, gg.auroc - 0.5, width * 0.92, bottom=0.5, color=SCORE_COLOR[s],
                   yerr=[gg.auroc - gg.lo, gg.hi - gg.auroc], capsize=2,
                   label={"n": "n = zᵀb (noise only)", "−h": "−h (imageness)", "A4D": "A⁴D (re-impl., 10 prompts)"}[s])
        ax.axhline(0.5, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(att)), att)
        ax.set(ylim=(0, 1), ylabel="AUROC (mean over classifiers)",
               title=f"({'ab'[cfg.datasets.index(dataset)]}) {dict(coco='COCO', tiny='Tiny-ImageNet')[dataset]}")
        legend_below(ax, ncol=3)
    save_figure(ctx.out, fig, "fig1_attack_auroc")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ref_clf = "resnet34" if "resnet34" in classifiers else classifiers[0]
    ds0 = cfg.datasets[0]
    sub = tab2[(tab2.dataset == ds0) & (tab2.classifier == ref_clf)]
    ax = axes[0]
    for r in sub.itertuples():
        ax.annotate("", xy=(r.adv_c_mean, r.adv_n_mean), xytext=(r.clean_c_mean, r.clean_n_mean),
                    arrowprops=dict(arrowstyle="->", color=ATTACK_COLOR[r.attack], lw=2))
        ax.plot([], [], color=ATTACK_COLOR[r.attack], label=r.attack)
    ax.scatter([sub.clean_c_mean.iloc[0]], [sub.clean_n_mean.iloc[0]], color=INK, s=30, zorder=5, label="clean mean")
    xs = np.r_[sub.clean_c_mean, sub.adv_c_mean]; ys = np.r_[sub.clean_n_mean, sub.adv_n_mean]
    pad = max(np.ptp(xs), np.ptp(ys), 1e-3) * 0.25
    ax.set(xlim=(xs.min() - pad, xs.max() + pad), ylim=(ys.min() - pad, ys.max() + pad))
    ax.set(xlabel=r"$c = z^\top\hat r$", ylabel=r"$n = z^\top b$", title=f"(a) Mean shift clean → attacked ({ds0}, {ref_clf})")
    legend_below(ax, ncol=3)
    ax = axes[1]; atts = list(sub.attack)
    data = [ctx.summary["_dh"][(ds0, ref_clf, a)] for a in atts]
    parts = ax.violinplot(data, positions=range(len(atts)), showmedians=True, showextrema=False)
    for body, a in zip(parts["bodies"], atts):
        body.set_facecolor(ATTACK_COLOR[a]); body.set_alpha(0.4)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(len(atts)), atts, rotation=25)
    ax.set(ylabel="Δh (attacked − clean, per image)", title="(b) Imageness change per attack")
    ax = axes[2]
    for dataset, col in zip(cfg.datasets, (C1, C2)):
        for s, ls, mk in (("n", ":", "s"), ("−h", "-", "o"), ("A4D", "--", "^")):
            ss = tab_s[(tab_s.dataset == dataset) & (tab_s.score == s)]
            band(ax, ss.eps_255, ss.auroc, ss.ci_low, ss.ci_high, col, f"{dataset}: {s}", ls=ls, marker=mk, ms=4)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set(xlabel="PGD ε (×1/255)", ylabel="AUROC", title=f"(c) PGD budget sweep ({sweep_clf})", ylim=(0, 1))
    legend_below(ax, ncol=3)
    save_figure(ctx.out, fig, "fig2_attack_mechanism")


# =============================================================================
# P2. NIH 부정 프롬프트 (CARZero)
# =============================================================================
def _section_p2(ctx: an.Context, cfg: AppConfig) -> None:
    an._banner("P2", "NIH negation with 10 template pairs (CARZero)")
    enc = ctx.encoder("carzero")
    imgs = ctx.images("carzero", "nih"); Z = imgs["Z"]; labels = imgs["labels"]
    pos_texts, d_idx, l_idx = an.nih_prompts("pos"); neg_texts, _, _ = an.nih_prompts("neg")
    Tp = normalize(ctx.cached("txt_carzero_nih_pos", dict(kind="nih_pos_templates", encoder=enc.signature, texts=pos_texts),
                              lambda: enc.encode_texts(pos_texts)))
    Tn = normalize(ctx.cached("txt_carzero_nih_neg", dict(kind="nih_neg_templates", encoder=enc.signature, texts=neg_texts),
                              lambda: enc.encode_texts(neg_texts)))
    AI = Axes(Z.mean(0), ctx.pooled_b("carzero", "image"))
    AT = Axes(np.concatenate([Tp, Tn]).mean(0), ctx.pooled_b("carzero", "text"))
    dT = Tp - Tn; v = Tn - Tp
    D = Z - AI.r_I; db, de = D @ AI.b, D @ AI.e
    Pz = {"orig": Z, "rm_b": normalize(AI.project_out(Z, "b")), "rm_be": normalize(AI.project_out(Z, "be"))}
    Pd = {"orig": dT, "rm_b": AI.project_out(dT, "b"), "rm_be": AI.project_out(dT, "be")}
    Z_rm = dict(b=Pz["rm_b"], be=Pz["rm_be"])
    def rm_text(T):
        return normalize(T - np.outer(T @ AT.b, AT.b))
    Tp_rm, Tn_rm = rm_text(Tp), rm_text(Tn)
    rows = []
    for j in range(len(dT)):
        y = labels[:, d_idx[j]].astype(bool)
        g = Z @ dT[j]
        M = float(AI.r_I @ dT[j]); V = db * (dT[j] @ AI.b); Aa = de * (dT[j] @ AI.e); U = g - M - V - Aa
        if np.max(np.abs(U - AI.project_out(D, "be") @ AI.project_out(dT[j:j + 1], "be")[0])) > 1e-6:
            raise AssertionError("Four-term reconstruction failed.")
        Dn = abs(M) + np.abs(V).mean() + np.abs(Aa).mean() + np.abs(U).mean() + 1e-10
        row = dict(disease=NIH_CLASSES[d_idx[j]], template=int(l_idx[j]), positive=pos_texts[j], negative=neg_texts[j],
                   auc=auroc(g[y], g[~y]), mean_g=float(g.mean()),
                   M=M, frac_M=abs(M) / Dn, frac_V=float(np.abs(V).mean() / Dn), frac_A=float(np.abs(Aa).mean() / Dn),
                   frac_U=float(np.abs(U).mean() / Dn))
        for k in ("orig", "rm_b", "rm_be"):
            s = Pz[k] @ (Pd[k][j] / np.linalg.norm(Pd[k][j]))
            row[f"dmu_{k}"] = float(s[y].mean() - s[~y].mean())
        for name, (Zv, tp, tn) in dict(rm_b_img=(Z_rm["b"], Tp[j], Tn[j]), rm_be_img=(Z_rm["be"], Tp[j], Tn[j]),
                                       rm_bT_text=(Z, Tp_rm[j], Tn_rm[j]), rm_both=(Z_rm["b"], Tp_rm[j], Tn_rm[j])).items():
            gv = Zv @ tp - Zv @ tn; row[f"dauc_{name}"] = auroc(gv[y], gv[~y]) - row["auc"]
        nd = np.linalg.norm(dT[j]); nv = np.linalg.norm(v[j])
        row.update(cos_dt_bI=float(dT[j] @ AI.b / nd), cos_dt_eI=float(dT[j] @ AI.e / nd),
                   cos_dt_bT=float(dT[j] @ AT.b / nd), cos_dt_eT=float(dT[j] @ AT.e / nd),
                   resid_frac_img=float(np.linalg.norm(AI.project_out(dT[j:j + 1], "be")) / nd),
                   resid_frac_text=float(np.linalg.norm(AT.project_out(dT[j:j + 1], "be")) / nd),
                   cos_tpos_bT=float(Tp[j] @ AT.b), cos_tneg_bT=float(Tn[j] @ AT.b), v_bT=float(v[j] @ AT.b),
                   v_frac_bT=float(abs(v[j] @ AT.b) / nv), v_frac_eT=float(abs(v[j] @ AT.e) / nv),
                   v_frac_resid=float(np.linalg.norm(AT.project_out(v[j:j + 1], "be")) / nv))
        rows.append(row)
    tab3 = save_csv(ctx.out, "tab3_nih_negation_by_template", rows)
    rng = np.random.default_rng(2323); summ = []
    num = [c for c in tab3.columns if c not in ("disease", "template", "positive", "negative")]
    for d, sub in tab3.groupby("disease", sort=False):
        row = dict(disease=d, n_templates=len(sub), frac_templates_v_toward_bT=float((sub.v_bT > 0).mean()))
        for c in num:
            lo, hi = bootstrap_mean_ci(sub[c].to_numpy(), cfg.bootstrap_repeats, rng)
            row.update({f"{c}_mean": float(sub[c].mean()), f"{c}_ci_low": lo, f"{c}_ci_high": hi})
        summ.append(row)
    tab4 = save_csv(ctx.out, "tab4_nih_negation_by_disease", summ)
    corr = []
    for unit, f in (("disease_template (140)", tab3), ("disease (14, template mean)", tab3.groupby("disease", sort=False).mean(numeric_only=True))):
        for mtr in ("dmu_orig", "dmu_rm_b", "dmu_rm_be", "mean_g", "M", "cos_dt_bT", "cos_tneg_bT", "v_bT", "cos_dt_bI"):
            rho, p = stats.spearmanr(f[mtr], f["auc"])
            corr.append(dict(unit=unit, metric=mtr, rho=float(rho), p=float(p)))
    corr = pd.DataFrame(corr); corr["q_bh"] = np.concatenate([an.bh_qvalues(g.p.to_numpy()) for _, g in corr.groupby("unit", sort=False)])
    save_csv(ctx.out, "tab5_nih_negation_correlation", corr)
    ctx.summary["P2"] = dict(mean_auc=float(tab3.auc.mean()), frac_v_toward_bT=float((tab3.v_bT > 0).mean()),
                             rho_dmu_rm_be_disease=float(corr[(corr.unit.str.startswith("disease (")) & (corr.metric == "dmu_rm_be")].rho.iloc[0]),
                             cos_bI_bT=float(AI.b @ AT.b))
    print(f"  mean AUC = {tab3.auc.mean():.3f}; negation moves toward b_T in {(tab3.v_bT > 0).mean() * 100:.3f}% of 140 pairs")
    if not cfg.make_figures:
        return
    t4 = tab4.set_index("disease")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5))
    ax = axes[0]
    ax.scatter(tab3.cos_tpos_bT, tab3.cos_tneg_bT, s=14, color=C2, alpha=0.6, label="disease × template (140)")
    lim = [min(tab3.cos_tpos_bT.min(), tab3.cos_tneg_bT.min()) - 0.02, max(tab3.cos_tpos_bT.max(), tab3.cos_tneg_bT.max()) + 0.02]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=1, label="no change")
    ax.set(xlim=lim, ylim=lim, aspect="equal", xlabel=r"cos($t^+$, $b_T$)", ylabel=r"cos($t^-$, $b_T$)",
           title="(a) Negated prompts vs. the text noise point")
    legend_below(ax)
    ax = axes[1]; x = np.arange(len(t4)); bottom = np.zeros(len(t4))
    # 직교 성분의 제곱 비중을 먼저 계산하고 템플릿에 대해 평균한다.
    sq = {k: tab3.assign(squared=tab3[f'v_frac_{k}']**2).groupby('disease').squared.mean().reindex(t4.index).to_numpy()
          for k in ('bT','eT','resid')}
    for k, col, lab in (("bT", C2, r"$b_T$"), ("eT", C4, r"$e_T$"), ("resid", C1, "residual")):
        ax.bar(x, sq[k], bottom=bottom, color=col, label=lab, edgecolor="white", lw=0.5); bottom += sq[k]
    ax.set_xticks(x, t4.index, rotation=60, fontsize=7)
    ax.set(ylabel="Squared share of ‖v‖ (template mean)", title=r"(b) Negation movement $v = t^- - t^+$")
    legend_below(ax, ncol=3, offset=-0.42)
    ax = axes[2]
    ax.errorbar(t4.dmu_rm_be_mean, t4.auc_mean, xerr=[t4.dmu_rm_be_mean - t4.dmu_rm_be_ci_low, t4.dmu_rm_be_ci_high - t4.dmu_rm_be_mean],
                yerr=[t4.auc_mean - t4.auc_ci_low, t4.auc_ci_high - t4.auc_mean], fmt="o", color=C2, capsize=2)
    for name, r in t4.iterrows():
        ax.annotate(name[:5], (r.dmu_rm_be_mean, r.auc_mean), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set(xlabel="Δμ(g) in residual space (3-way)", ylabel="AUC",
           title=f"(c) Residual separation vs AUC (ρ = {ctx.summary['P2']['rho_dmu_rm_be_disease']:.3f})")
    save_figure(ctx.out, fig, "fig3_nih_negation")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    ax = axes[0]; bottom = np.zeros(len(t4))
    for term, col in (("M", C1), ("V", C2), ("A", C4), ("U", C3)):
        vals = t4[f"frac_{term}_mean"].to_numpy(); ax.bar(x, vals, bottom=bottom, color=col, label=term, edgecolor="white", lw=0.5)
        bottom += vals
    ax.set_xticks(x, t4.index, rotation=60, fontsize=7)
    ax.set(ylabel="Share of |M|+E|V|+E|A|+E|U|", title="(a) Four-term contribution of g = zᵀΔt")
    legend_below(ax, ncol=4, offset=-0.42)
    ax = axes[1]; width = 0.2
    names = dict(rm_b_img="−b_I (image)", rm_be_img="−b_I, −e_I (image)", rm_bT_text="−b_T (text)", rm_both="−b_I, −b_T")
    for k, (var, col) in enumerate(zip(names, (C1, C2, C3, C4))):
        m = t4[f"dauc_{var}_mean"]
        ax.bar(x + (k - 1.5) * width, m, width, color=col, label=names[var],
               yerr=[m - t4[f"dauc_{var}_ci_low"], t4[f"dauc_{var}_ci_high"] - m], capsize=1.5)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x, t4.index, rotation=60, fontsize=7)
    ax.set(ylabel="ΔAUC (mean over 10 templates)", title="(b) Component removal")
    legend_below(ax, ncol=4, offset=-0.42)
    save_figure(ctx.out, fig, "fig4_nih_negation_decomposition")


# =============================================================================
# Main
# =============================================================================



# =============================================================================
# 섹션 2: 세 실험. 함수마다 CSV/PNG 한 쌍.
# =============================================================================
@an.experiment('exp07_score_decomposition')
def exp07_score_decomposition(ctx):
    """긍정 점수의 M/V/A/U 비중, 분리도, AUC와의 관계, 성분 제거."""
    an._section_a9(ctx)


@an.experiment('exp08_negation_geometry')
def exp08_negation_geometry(ctx, cfg=None):
    """질환×부정 템플릿별 이동, 점수 차이의 분해, 잔차 분리도/성분 제거."""
    _section_p2(ctx, cfg or ctx.cfg)


@an.experiment('exp09_attack_detection')
def exp09_attack_detection(ctx, cfg=None):
    """기존 P1과 같은 조건의 탐지 + 동일 이미지의 PGD 예산별 좌표/거리.
    공격 예산별 출력은 각 최적화 결과의 비교이며 하나의 최적화 반복 경로가 아니다.
    """
    cfg=cfg or ctx.cfg
    _section_p1(ctx,cfg)
    clip=ctx.encoder('clip');rows=[];cross=[];curves=[]
    b=ctx.pooled_b('clip','image')
    classifier='resnet34' if 'resnet34' in cfg.classifiers else cfg.classifiers[0]
    other_vertices={f:normalize(an.matched_noise_embeddings(ctx,'clip',f,cfg.seeds[0]).mean(0)) for f in an.MATCHED_FAMILIES}
    for dataset in cfg.datasets:
        paths=dataset_paths(cfg,ctx,dataset)
        perm=np.random.default_rng(4040).permutation(len(paths))
        ref=[paths[j] for j in perm[:cfg.n_reference_clean]]
        test=[paths[j] for j in perm[cfg.n_reference_clean:cfg.n_reference_clean+cfg.n_attack_images]]
        Zref=normalize(clip.encode_pixels(attack_pixels(p) for p in ref))
        Zclean=normalize(clip.encode_pixels(attack_pixels(p) for p in test))
        A=Axes(Zref.mean(0),b);S0=A.coords(Zclean)
        budgets=(0.000,)+tuple(sorted(set(cfg.pgd_eps_sweep)))
        states=[Zclean]
        for eps in budgets[1:]:
            res=run_attack(ctx,cfg,dataset,classifier,'PGD',eps,test,f'n{cfg.n_attack_images}')
            states.append(normalize(res['Z']))
        Z=np.stack(states,axis=1);T=an.trajectory_metrics(Z,b)
        for i,p in enumerate(test):
            curves.append(dict(dataset=dataset,classifier=classifier,image=p.name,
                               budget_curve_length=float(T['path_length'][i]),
                               displacement=float(T['displacement'][i]),straightness=float(T['straightness'][i])))
        for li,eps in enumerate(budgets):
            S=A.coords(states[li])
            for i,p in enumerate(test):
                rows.append(dict(dataset=dataset,classifier=classifier,image=p.name,eps_255=eps*255,
                                 c=float(S['c'][i]),h=float(S['h'][i]),n=float(S['n'][i]),
                                 delta_c=float(S['c'][i]-S0['c'][i]),delta_h=float(S['h'][i]-S0['h'][i]),
                                 delta_n=float(S['n'][i]-S0['n'][i]),distance_to_b=float(T['distance'][i,li])))
            for reference,vertex in {'uniform_rgb':b,**other_vertices}.items():
                cos=states[li] @ vertex;baseline=Zclean @ vertex
                cross.append(dict(dataset=dataset,classifier=classifier,reference_family=reference,eps_255=eps*255,
                                  distance_mean=float(np.sqrt(np.maximum(0,2-2*cos)).mean()),
                                  delta_n_mean=float((cos-baseline).mean())))
    df=save_csv(ctx.out,'pgd_budget_coordinates',rows)
    save_csv(ctx.out,'pgd_budget_curve_metrics',curves)
    cr=save_csv(ctx.out,'pgd_cross_vertex_approach',cross)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for dataset,d in df.groupby('dataset'):
        g=d.groupby('eps_255').mean(numeric_only=True)
        axs[0].plot(g.index,g.delta_h,marker='o',label=dataset)
        axs[1].plot(g.index,g.distance_to_b,marker='o',label=dataset)
    for (dataset,reference),d in cr.groupby(['dataset','reference_family']):
        axs[2].plot(d.eps_255,d.delta_n_mean,label=f'{dataset}/{reference}')
    axs[0].set(xlabel='PGD budget (×1/255)',ylabel='Imageness change');axs[0].legend()
    axs[1].set(xlabel='PGD budget (×1/255)',ylabel='Distance to uniform RGB vertex')
    axs[2].set(xlabel='PGD budget (×1/255)',ylabel='Change in reference cosine');axs[2].legend(fontsize=6)
    save_figure(ctx.out,fig,'pgd_budget_geometry')


EXPERIMENTS=(exp07_score_decomposition,exp08_negation_geometry,exp09_attack_detection)


def run_application(config=APP,encoders=None):
    ctx=an.Context(config,encoders,out_name='application')
    try:
        if 'carzero' in config.encoders or config.run_p2:ctx.data.nih
        if config.run_p1:
            for dataset in config.datasets:
                paths=dataset_paths(config,ctx,dataset)
                if len(paths)<config.n_reference_clean+config.n_attack_images:
                    raise ValueError(f'{dataset}: 기준/공격 이미지가 설정한 개수보다 적습니다.')
        exp07_score_decomposition(ctx)
        if config.run_p2:exp08_negation_geometry(ctx,config)
        if config.run_p1:exp09_attack_detection(ctx,config)
    finally:
        for encoder in ctx._encoders.values():
            if hasattr(encoder,'close'):encoder.close()
    print(f'완료: {ctx.out}',flush=True)
    return ctx.out
