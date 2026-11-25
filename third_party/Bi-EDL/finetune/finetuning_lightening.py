from transformers import AutoTokenizer
import torch
from pytorch_lightning.core import LightningModule
from torchmetrics.classification import MultilabelAUROC
import CARZero.builder as builder
import CARZero
import torch.nn.functional as F
from typing import Tuple


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) where Dir(1) is uniform over K classes.
    alpha: (N, K), alpha > 0
    return: (N,) KL per sample
    """
    device = alpha.device
    N, K = alpha.shape
    beta = torch.ones((1, K), device=device, dtype=alpha.dtype).expand_as(alpha)  # (N, K)

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)  # (N, 1)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)    # (N, 1) == K

    # ln B(alpha) = sum lgamma(alpha_i) - lgamma(sum alpha)
    lnB_alpha = torch.sum(torch.lgamma(alpha), dim=1, keepdim=True) - torch.lgamma(sum_alpha)
    lnB_beta  = torch.sum(torch.lgamma(beta),  dim=1, keepdim=True) - torch.lgamma(sum_beta)

    # KL = lnB(beta) - lnB(alpha) + sum (alpha_i - beta_i) * (digamma(alpha_i) - digamma(sum_alpha))
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)

    kl = (lnB_beta - lnB_alpha) + torch.sum(
        (alpha - beta) * (digamma_alpha - digamma_sum_alpha),
        dim=1,
        keepdim=True
    )  # (N, 1)

    return kl.squeeze(1)  # (N,)


class MCQEDLLightModel(LightningModule):
    def __init__(self, cfg, backbone_model=None):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.backbone_model = backbone_model
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.neg_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.failure_auroc_metric = MultilabelAUROC(num_labels=14, average=None)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']

        self.pos_prompts = [f"There is {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]
        self.neg_prompts = [f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]]

        self.prompts = [*self.pos_prompts, *self.neg_prompts]

        self.toks = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        self.cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in self.toks["input_ids"]],
            dtype=torch.long
        )

    def setup(self, stage=None):
        if self.backbone_model is None:
            self.backbone_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=None, multi=self.cfg.model.CARZero.multi, cfg=self.cfg)
            self.freeze_module()
            self.print("CARZero model loaded and frozen.")
        if self.cfg.peft.enabled :
            self.print("Setting up PEFT for the student model...")
            self.set_peft()
        if self.dm is None:
            self.dm = self.trainer.datamodule

    def set_peft(self):
        r = self.cfg.peft.r
        alpha = self.cfg.peft.alpha
        dropout = self.cfg.peft.dropout
        adaptor_name = self.cfg.peft.adaptor_name

        self.print(f"Setting up PEFT with r={r}, alpha={alpha}, dropout={dropout}, adaptor_name={adaptor_name}")

    def freeze_module(self):
        freeze_dict = getattr(self.cfg, "freeze", {})
        if freeze_dict.get("image", False):
            for param in self.backbone_model.img_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("text", False):
            for param in self.backbone_model.text_encoder.parameters():
                param.requires_grad = False
        if freeze_dict.get("fusion", False):
            if self.cfg.model.CARZero.multi == False:
                for param in self.backbone_model.fusion_module.parameters():
                    param.requires_grad = False
            else :
                for param in self.backbone_model.i2t_fusion_module.parameters():
                    param.requires_grad = False
                for param in self.backbone_model.t2i_fusion_module.parameters():
                    param.requires_grad = False

        self.print("==== Frozen Modules ====")
        self.print(" -> image encoder frozen:", all(not p.requires_grad for p in self.backbone_model.img_encoder.parameters()))
        self.print(" -> text encoder frozen:", all(not p.requires_grad for p in self.backbone_model.text_encoder.parameters()))

        if self.cfg.model.CARZero.multi == False:
            self.print(" -> fusion module frozen:", all(not p.requires_grad for p in self.backbone_model.fusion_module.parameters()))
        else :
            self.print(" -> i2t fusion module frozen:", all(not p.requires_grad for p in self.backbone_model.i2t_fusion_module.parameters()))
            self.print(" -> t2i fusion module frozen:", all(not p.requires_grad for p in self.backbone_model.t2i_fusion_module.parameters()))

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.backbone_model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def generate_i2t_mcq(self, labels_batch: torch.Tensor, shuffle=True) -> Tuple[torch.Tensor, torch.Tensor]:
        B = labels_batch.shape[0]
        num_diseases = labels_batch.shape[1] - 1
        device = labels_batch.device

        disease_labels = labels_batch[:, :num_diseases] # (B, 14)

        pos_idx_range = torch.arange(num_diseases).to(device)
        neg_idx_range = torch.arange(num_diseases, 2 * num_diseases).to(device)

        # 2. 정답 선택 로직 (기존 로직 이식)
        # 각 샘플별로 긍정문 정답이 존재하는지 확인 (Any)
        has_pos_true = (disease_labels == 1).any(dim=1) # (B,)
        has_neg_true = (disease_labels == 0).any(dim=1)
        # 50% 확률로 긍정문을 정답으로 쓸지 결정
        use_pos_preference = torch.rand(B).to(device) < 0.5
        # 실제로 긍정문을 정답으로 선택할 샘플들
        select_pos = (has_pos_true & use_pos_preference) | (~has_neg_true)

        # 3. 정답 인덱스 추출 (행별로 다름)
        # 긍정 정답을 쓸 샘플은 1인 위치에서, 나머지는 0인 위치(부정 정답)에서 랜덤하게 하나 선택
        # 이를 위해 각 샘플의 정답 후보들에 가중치를 주어 multinomial로 뽑습니다.
        weights = torch.where(select_pos.unsqueeze(1),
                            (disease_labels == 1).float(),
                            (disease_labels == 0).float())

        # 각 행에서 가중치가 있는 곳 중 하나를 랜덤 추출 (정답의 질환 번호)
        selected_disease_idx = torch.multinomial(weights, 1).squeeze(1) # (B,)

        # 실제 프롬프트 인덱스로 변환
        # select_pos인 행은 긍정문(0~13), 아니면 부정문(14~27) 인덱스 선택
        answer_indices = torch.where(select_pos,
                                    selected_disease_idx,
                                    selected_disease_idx + num_diseases)

        # 4. 오답(Wrong) 선택 (2개)
        # 오답 후보: 정답과 반대되는 성격의 문장들
        # (label=1이면 부정문이 오답, label=0이면 긍정문이 오답)
        false_matrix = torch.where(disease_labels == 1, neg_idx_range, pos_idx_range)

        # 각 행에서 오답 후보 14개 중 2개를 랜덤 추출
        wrong_offsets = torch.multinomial(torch.ones((B, num_diseases)).to(device), 2, replacement=False)
        wrong_indices = false_matrix.gather(1, wrong_offsets)

        # 5. 최종 구성 및 셔플
        choices = torch.cat([answer_indices.unsqueeze(1), wrong_indices], dim=1)

        if shuffle:
            shuffled_idx = torch.argsort(torch.rand(B, 3).to(device), dim=1)
            choices = choices.gather(1, shuffled_idx)
            targets = (choices == answer_indices.unsqueeze(1)).nonzero()[:, 1]
        else:
            targets = torch.zeros(B, dtype=torch.long, device=device)

        return choices, targets

    def generate_t2i_mcq(self,labels_batch: torch.Tensor, shuffle=True):
        """
        labels_batch: (B, 15) - 배치 내 이미지들의 레이블
        반환값:
            valid_prompt_indices: (N,) - 유효한(정답이 존재하는) 프롬프트 번호들
            image_choices: (N, 3) - 각 유효 프롬프트별 선택된 이미지 인덱스 [Ans, W1, W2]
            targets: (N,) - 3개 중 정답 위치
        """
        B = labels_batch.shape[0]
        num_diseases = 14
        device = labels_batch.device

        # 1. 28개 프롬프트에 대한 전체 정답 지도 생성 (28, B)
        # 0~13: Positive (label 1), 14~27: Negative (label 0)
        disease_labels = labels_batch[:, :num_diseases].T  # (14, B)

        # row 0~13: 긍정문 일치 여부, row 14~27: 부정문 일치 여부
        is_correct_map = torch.cat([
            (disease_labels == 1),  # Positive Prompts
            (disease_labels == 0)   # Negative Prompts
        ], dim=0) # (28, B)

        # 2. 유효한 프롬프트 필터링
        # 정답(True)이 하나 이상 있고, 오답(False)이 두 개 이상 있는 프롬프트만 선택
        has_ans = is_correct_map.any(dim=1)
        has_wrongs = (~is_correct_map).sum(dim=1) >= 2
        valid_mask = has_ans & has_wrongs # (28,)

        valid_prompt_indices = torch.where(valid_mask)[0]
        num_valid = valid_prompt_indices.size(0)

        if num_valid == 0:
            return None, None, None

        # 유효한 프롬프트에 대한 맵만 추출
        filtered_map = is_correct_map[valid_prompt_indices] # (num_valid, B)

        # 3. 이미지 선택 (모든 이미지가 최대한 활용되도록 가중치 부여 가능)
        # 여기서는 각 프롬프트별로 독립적으로 샘플링하되, multinomial을 통해 무작위성 확보
        ans_weights = filtered_map.float()
        wrong_weights = (~filtered_map).float()

        # 정답 이미지 1개씩 추출 (num_valid, 1)
        ans_img_idx = torch.multinomial(ans_weights, 1)

        # 오답 이미지 2개씩 추출 (num_valid, 2)
        wrong_img_indices = torch.multinomial(wrong_weights, 2, replacement=False)

        # 4. 최종 구성 및 셔플
        image_choices = torch.cat([ans_img_idx, wrong_img_indices], dim=1) # (num_valid, 3)

        if shuffle:
            # (num_valid, 3) 크기의 셔플 인덱스 생성
            shuffled_idx = torch.argsort(torch.rand(num_valid, 3).to(device), dim=1)
            image_choices = image_choices.gather(1, shuffled_idx)
            # 정답이 이동한 위치 추적
            targets = (image_choices == ans_img_idx).nonzero()[:, 1]
        else:
            targets = torch.zeros(num_valid, dtype=torch.long, device=device)

        return valid_prompt_indices, image_choices, targets

    def image_forward(self, batch) :
        imgs = batch["imgs"].to(self.device)
        img_emb_l, img_emb_g = self.backbone_model.image_encoder_forward(imgs)
        return img_emb_l, img_emb_g

    def text_forward(self) :
        text_emb_l, text_emb_g, sents = self.backbone_model.text_encoder_forward(self.toks['input_ids'].to(self.device),
                                                                                self.toks['attention_mask'].to(self.device),
                                                                                self.toks['token_type_ids'].to(self.device),
                                                                                )
        return text_emb_l, text_emb_g, sents

    def fusion_forward(self, img_l, img_g, txt_l, txt_g):
        """
        img_l: (B, D, H, W) - 이미지 로컬 특징
        img_g: (B, D)       - 이미지 글로벌 특징
        txt_l: (T, D, L)    - 텍스트 로컬 특징 (L: 토큰 길이)
        txt_g: (T, D)       - 텍스트 글로벌 특징
        """
        B = img_l.shape[0]
        T = txt_g.shape[0]
        D = img_g.shape[1]

        # 1. Local feature 준비 (Flatten & Permute)
        # img_l: (B, D, H*W) -> (B, S_img, D)
        img_l_flat = img_l.view(B, D, -1).permute(0, 2, 1)
        # txt_l: (T, D, L) -> (T, S_txt, D)
        txt_l_flat = txt_l.permute(0, 2, 1)

        # 2. B x T 조합을 위한 확장 (Expansion)
        # (B, 1, S_img, D) -> (B, T, S_img, D) -> (B*T, S_img, D)
        img_l_exp = img_l_flat.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, D)
        # (1, T, S_txt, D) -> (B, T, S_txt, D) -> (B*T, S_txt, D)
        txt_l_exp = txt_l_flat.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * T, -1, D)

        # 3. Global Query 준비 (B*T 개로 복사 및 차원 조정)
        # 이미지 쿼리: (B, 1, D) -> (B, T, D) -> (1, B*T, D)
        img_g_q = img_g.unsqueeze(1).expand(-1, T, -1).reshape(1, B * T, D)
        # 텍스트 쿼리: (1, T, D) -> (B, T, D) -> (1, B*T, D)
        txt_g_q = txt_g.unsqueeze(0).expand(B, -1, -1).reshape(1, B * T, D)

        # 4. Fusion Module을 통한 유사도 계산
        # I2T: 이미지를 쿼리로 텍스트의 로컬 토큰들을 참조
        i2t_logit = self.backbone_model.i2t_fusion_module(
            txt_l_exp, img_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)

        # T2I: 텍스트를 쿼리로 이미지의 로컬 패치들을 참조
        t2i_logit = self.backbone_model.t2i_fusion_module(
            img_l_exp, txt_g_q, inside_repeat=False
        ).squeeze(-1).squeeze(-1) # (B*T)

        # 5. (B, T) 유사도 행렬로 재구성
        i2t_matrix = i2t_logit.view(B, T)
        t2i_matrix = t2i_logit.view(B, T)

        return i2t_matrix, t2i_matrix

    def i2t_forward(self, i2t_cls, labels):
        i2t_choices, i2t_targets = self.generate_i2t_mcq(labels.to(self.device))

        i2t_logits = i2t_cls.gather(1, i2t_choices.to(self.device)) # (N, 3)
        i2t_ce_loss = F.cross_entropy(i2t_logits, i2t_targets.to(self.device), reduction='mean')
        i2t_acc = (i2t_logits.argmax(dim=1) == i2t_targets.to(self.device)).float().mean()

        return i2t_ce_loss, i2t_acc

    def t2i_forward(self, t2i_cls, labels):
        t2i_prompt_indices, t2i_image_choices, t2i_targets = self.generate_t2i_mcq(labels.to(self.device))

        if t2i_prompt_indices is None:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, torch.tensor(0.0, device=self.device)

        t2i_logits = t2i_cls.T[t2i_prompt_indices].gather(1, t2i_image_choices.to(self.device)) # (T, 3)
        t2i_ce_loss = F.cross_entropy(t2i_logits, t2i_targets.to(self.device), reduction='mean')
        t2i_acc = (t2i_logits.argmax(dim=1) == t2i_targets.to(self.device)).float().mean()

        return t2i_ce_loss, t2i_acc

    def edl_forward(self, i2t_cls, t2i_cls, labels):
        """
        i2t_cls, t2i_cls: (B, T) - T=28 (0~13: Pos, 14~27: Neg)
        labels: (B, 15) - 14개 질환 + 1개 No Finding
        """
        B = i2t_cls.size(0)
        num_diseases = 14

        # 1. 타겟 질환(0~13) 결정
        # 각 샘플이 가진 질환들(label=1) 중 하나를 선택
        disease_labels = labels[:, :num_diseases] # (B, 14)
        nf_mask = labels[:, -1] == 1 # No Finding 샘플 마스크 (B,)

        # 질환이 있는 샘플은 있는 것 중 하나, NF 샘플은 전체(14개) 중 하나 무작위 선택
        # weights: 질환이 있으면 해당 위치 1, NF면 모든 위치 1
        selection_weights = disease_labels.float()
        selection_weights[nf_mask] = 1.0

        # target_disease_idx: (B,) 각 이미지당 비교할 질환 번호 (0~13)
        target_disease_idx = torch.multinomial(selection_weights, 1).squeeze(1)

        # 2. 긍정 vs 부정 로짓 추출
        # i2t와 t2i 평균 점수 계산 (B, T)
        avg_logits = (i2t_cls / torch.exp(self.backbone_model.i2t_tau) +
                    t2i_cls / torch.exp(self.backbone_model.t2i_tau)) / 2

        # 동일 질환의 긍정(0~13) 및 부정(14~27) 인덱스
        pos_prompts_idx = target_disease_idx
        neg_prompts_idx = target_disease_idx + num_diseases

        pos_scores = avg_logits[torch.arange(B), pos_prompts_idx] # (B,)
        neg_scores = avg_logits[torch.arange(B), neg_prompts_idx] # (B,)

        # 3. 정답(Ground Truth) 설정
        # 질환이 있는 샘플(NF가 아님)은 긍정이 정답(0), NF 샘플은 부정이 정답(1)
        # beta_logits: [Pos_Score, Neg_Score] (B, 2)
        beta_logits = torch.stack([pos_scores, neg_scores], dim=1)

        # targets: 0 이면 긍정 프롬프트가 참, 1 이면 부정 프롬프트가 참
        # NF 이미지가 아니면(질환 유) 0, NF 이미지면 1
        beta_targets = nf_mask.long()

        # 4. EDL Loss 계산
        alpha = F.softplus(beta_logits) + 1.0 # (B, 2)
        S = torch.sum(alpha, dim=1, keepdim=True)

        # 정답 클래스에 해당하는 alpha 값 선택
        alpha_y = alpha.gather(1, beta_targets.unsqueeze(1)) # (B, 1)

        pred = torch.argmax(beta_logits, dim=1)
        correct = (pred == beta_targets).float()
        acc = correct.mean()

        # Loss Match: 정답에 대한 증거 극대화
        loss_match = (torch.digamma(S) - torch.digamma(alpha_y)).squeeze(1).mean()

        # Loss KL: 불확실성 정규화
        y = F.one_hot(beta_targets, num_classes=2).to(alpha.dtype)
        tilde_alpha = y + (1.0 - y) * alpha
        loss_kl = dirichlet_kl_to_uniform(tilde_alpha).mean()

        loss_edl = loss_match + self.cfg.train.edl_weight * loss_kl
        return loss_edl, acc

    def shared_step(self, batch, split):
        img_l, img_g = self.image_forward(batch)
        txt_l, txt_g, sents = self.text_forward()

        i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)

        i2t_loss, i2t_acc = self.i2t_forward(i2t_cls, batch["label"])

        t2i_loss, t2i_acc = self.t2i_forward(t2i_cls, batch["label"])

        edl_loss, edl_acc = self.edl_forward(i2t_cls, t2i_cls, batch["label"])

        weight = self.cfg.train.weight

        epoch = self.current_epoch + 1
        lam = min(1.0, float(epoch) / self.cfg.train.lam)
        lam = torch.tensor(lam, device=self.device, dtype=edl_loss.dtype)

        loss = (1-lam)*(weight * i2t_loss + (1 - weight) * t2i_loss)+(lam * edl_loss)

        self.log_dict({f"{split}/loss": loss,
                       f"{split}/i2t_loss": i2t_loss,
                       f"{split}/t2i_loss": t2i_loss,
                       f"{split}/edl_loss": edl_loss,
                       f"{split}/i2t_acc": i2t_acc,
                       f"{split}/t2i_acc": t2i_acc,
                       f"{split}/edl_acc": edl_acc},
                  prog_bar=True, on_epoch=True)
        return loss

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

    def inference(self, batch, split="val") :
        with torch.no_grad():
            img_l, img_g = self.image_forward(batch)
            txt_l, txt_g, sents = self.text_forward()

            i2t_cls, t2i_cls = self.fusion_forward(img_l, img_g, txt_l, txt_g)

            avg_logits = (i2t_cls / torch.exp(self.backbone_model.i2t_tau) +
                t2i_cls / torch.exp(self.backbone_model.t2i_tau)) / 2
        return avg_logits

    def metrics(self, batch, split):
        labels = batch["label"].to(self.device)         # 멀티라벨 (1=질환 존재)

        all_logits = self.inference(batch, split=split)

        pos_logits = all_logits[:,:14]
        neg_logits = all_logits[:,14:]
        alpha_pos = F.softplus(pos_logits) + 1
        alpha_neg = F.softplus(neg_logits) + 1

        S = alpha_pos + alpha_neg

        pos_probs  = alpha_pos / S                             # 질환 존재 확률
        neg_probs  = alpha_neg / S                             # 질환 부재 확률

        U = 2 / S # (N, 14)

        U_mean = U.mean(dim=0)                                 # 클래스별 평균 불확실성

        targets = labels[:,:-1].int()                             # 질환 존재 → 1
        neg_targets = (1 - labels[:,:-1]).int()                # 질환 부재 → 1

        probs = pos_probs > neg_probs

        failure_case = (probs.int() != targets).int()
        self.failure_auroc_metric.update(U, failure_case)

        acc = (probs.int() == targets).float().mean()

        # ---------- 메트릭 누적 ----------
        self.auroc_metric.update(pos_probs,  targets)
        self.neg_auroc_metric.update(neg_probs, neg_targets)

        # ---------- BCE 손실 (positive-prompt 기준) ----------
        pos_bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets.float())
        neg_bce_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets.float())

        #bce_loss = 0.5 * (pos_bce_loss + neg_bce_loss)
        weight = self.cfg.train.weight
        bce_loss = weight*pos_bce_loss+(1-weight)*neg_bce_loss

        # ---------- 로깅 ----------
        self.log(f"{split}/bce_loss",       bce_loss,     prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f"{split}/acc", acc, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f"{split}/U_mean", U_mean.mean(), prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)

        self.log_dict({f"{split}/Uncertainty_{c}":     U_mean[i]
                          for i, c in enumerate(self.class_names[:-1])}, sync_dist=True, on_epoch=True, on_step=False)

        return bce_loss

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            self.print("==== Validation Epoch End ====")
            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            class_auroc = self.auroc_metric.compute()
            neg_class_auroc = self.neg_auroc_metric.compute()
            failure_class_auroc = self.failure_auroc_metric.compute()
            pos_mean_auroc = class_auroc.mean()
            neg_mean_auroc = neg_class_auroc.mean()
            mean_auroc = (pos_mean_auroc + neg_mean_auroc)/2
            failure_mean_auroc = failure_class_auroc.mean()

            self.print(f"[VAL] Epoch {self.current_epoch} Summary:")
            self.print(f" - val/mean_auroc: {mean_auroc.item():.4f}")
            self.print(f" - val/pos_mean_auroc: {pos_mean_auroc.item():.4f}")
            self.print(f" - val/neg_mean_auroc: {neg_mean_auroc.item():.4f}")
            self.print(f" - val/FD_mean_auroc: {failure_mean_auroc.item():.4f}")

            self.log(f"val/mean_auroc", mean_auroc, sync_dist=True)
            self.log(f"val/pos_mean_auroc", pos_mean_auroc, sync_dist=True)
            self.log(f"val/neg_mean_auroc", neg_mean_auroc, sync_dist=True)
            self.log(f"val/FD_mean_auroc", failure_mean_auroc, sync_dist=True)

            self.log_dict({f"val/auroc_{c}":     class_auroc[i]
                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
            self.log_dict({f"val/neg_auroc_{c}": neg_class_auroc[i]
                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)
            self.log_dict({f"val/FD_auroc_{c}": failure_class_auroc[i]
                           for i, c in enumerate(self.class_names[:-1])}, sync_dist=True)

            # 클래스별 AUROC만 따로 정렬 출력
            self.print(" - Class-wise AUROC:")
            for i, c in enumerate(self.class_names[:-1]):
                self.print(f"    {c:<22}: {class_auroc[i].item():.4f}")

            self.print(" - Negative Class-wise AUROC:")
            for i, c in enumerate(self.class_names[:-1]):
                self.print(f"    {c:<22}: {neg_class_auroc[i].item():.4f}")

            self.print(f" - Failure Detection Mean AUROC")
            for i, c in enumerate(self.class_names[:-1]):
                self.print(f"    {c:<22}: {failure_class_auroc[i].item():.4f}")

        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        self.failure_auroc_metric.reset()

    def on_test_epoch_end(self):
        class_auroc = self.auroc_metric.compute()
        mean_auroc = torch.mean(class_auroc)

        if self.trainer.is_global_zero:
            self.print(f"[TEST] Epoch Summary:")
            self.print(f" - test/pos_mean_auroc : {mean_auroc.item():.4f}")
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {class_auroc[i].item():.4f}")

            self.print(f" - test/neg_mean_auroc : {self.neg_auroc_metric.compute().mean().item():.4f}")
            neg_class_auroc = self.neg_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {neg_class_auroc[i].item():.4f}")

            self.print(f" - test/FD_mean_auroc : {self.failure_auroc_metric.compute().mean().item():.4f}")
            failure_class_auroc = self.failure_auroc_metric.compute()
            for i, cls in enumerate(self.class_names[:-1]):
                self.print(f"   {cls:<22}: {failure_class_auroc[i].item():.4f}")

        # metric 초기화 (다음 test run 대비)
        self.auroc_metric.reset()
        self.neg_auroc_metric.reset()
        self.failure_auroc_metric.reset()