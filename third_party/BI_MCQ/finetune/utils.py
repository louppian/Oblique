import random, secrets
_R = secrets.SystemRandom()
from typing import List, Tuple, Any, Dict, Optional, Set
import torch

def _pos_sent(disease: str) -> str:
    return f"There is {disease.replace('_', ' ')}."

def _neg_sent(disease: str) -> str:
    return f"There is no {disease.replace('_', ' ')}."

def _hyb_sent(pos: str, neg: str) -> str:
    return f"There is {pos.replace('_', ' ')} but no {neg.replace('_', ' ')}."

def generate_mcq(labels: List[int], class_names: List[str], shuffle=True, no_hyb=False) -> Tuple[List[str], int]:    
    NF = len(class_names) - 1
    pos_idx = [i for i, y in enumerate(labels) if y == 1 and i != NF]
    neg_idx = [i for i, y in enumerate(labels) if y == 0 and i != NF]

    if no_hyb:
        if labels[NF] == 1 or not pos_idx:
            neg_class = _R.choice(range(NF))
            pos_sent = _pos_sent(class_names[neg_class])
            neg_sent = _neg_sent(class_names[neg_class])

            choices = [pos_sent, neg_sent]
            if shuffle:
                _R.shuffle(choices)
            return choices, choices.index(neg_sent)
        ans_type = _R.choice(["POS", "NEG"])

        if ans_type == "POS":
            answer = _pos_sent(class_names[_R.choice(pos_idx)])
            wrong = _neg_sent(class_names[_R.choice(pos_idx)])
        else:
            answer = _neg_sent(class_names[_R.choice(neg_idx)])
            wrong = _pos_sent(class_names[_R.choice(neg_idx)])

        choices = [answer, wrong]
        if shuffle:
            _R.shuffle(choices)
        return choices, choices.index(answer)

    if labels[NF] == 1 or not pos_idx:
        pos_sent = _pos_sent(class_names[_R.choice(range(NF))])
        neg_sent = _neg_sent(class_names[_R.choice(range(NF))])
        hyb_sent = _hyb_sent(class_names[_R.choice(range(NF))], class_names[_R.choice(range(NF))])
        choices  = [pos_sent, neg_sent, hyb_sent]
        if shuffle:
            _R.shuffle(choices)
        return choices, choices.index(neg_sent)

    ans_type = _R.choice(["POS", "NEG", "HYB"])

    if ans_type == "POS":
        pos_sent = _pos_sent(class_names[_R.choice(pos_idx)])
        neg_sent = _neg_sent(class_names[_R.choice(pos_idx)])
        hyb_sent = _hyb_sent(class_names[_R.choice(neg_idx)], class_names[_R.choice(pos_idx)])
        answer  = pos_sent

    elif ans_type == "NEG":
        pos_sent = _pos_sent(class_names[_R.choice(neg_idx)])
        neg_sent = _neg_sent(class_names[_R.choice(neg_idx)])
        hyb_sent = _hyb_sent(class_names[_R.choice(neg_idx)], class_names[_R.choice(pos_idx)])
        answer  = neg_sent

    else:
        pos_sent = _pos_sent(class_names[_R.choice(neg_idx)])
        neg_sent = _neg_sent(class_names[_R.choice(pos_idx)])
        hyb_sent = _hyb_sent(class_names[_R.choice(pos_idx)], class_names[_R.choice(neg_idx)])
        answer  = hyb_sent

    choices = [pos_sent, neg_sent, hyb_sent]
    if shuffle:
        _R.shuffle(choices)
    return choices, choices.index(answer)

def generate_mcq2(labels: List[int], class_names: List[str], shuffle=True, no_hyb=False) -> Tuple[List[str], int]:
    NF = len(class_names) - 1
    pos_idx = [i for i, y in enumerate(labels) if y == 1 and i != NF]
    neg_idx = [i for i, y in enumerate(labels) if y == 0 and i != NF]

    def _rand_disease():
        return class_names[_R.choice(range(NF))]

    def _neg_or_nofinding(ratio=0.5, disease=None):
        if _R.random() < ratio:
            return _neg_sent(disease if disease is not None else _rand_disease())
        else:
            return "There is no finding."
    
    def _make_hyb_false(class_names, pos_idx, neg_idx):
        if len(pos_idx) >= 2:
            p1 = _R.choice(pos_idx)
            p2_candidates = [j for j in pos_idx if j != p1]
            p2 = _R.choice(p2_candidates)
            return _hyb_sent(class_names[p1], class_names[p2])

        patterns = []

        if len(neg_idx) >= 1 and len(pos_idx) >= 1:
            patterns.append("N_not_P")

        if len(neg_idx) >= 2:
            patterns.append("N_not_N")

        pattern_type = _R.choice(patterns)

        if pattern_type == "N_not_P":
            n = class_names[_R.choice(neg_idx)]
            p = class_names[_R.choice(pos_idx)]
            return _hyb_sent(n, p)

        else:
            n1 = _R.choice(neg_idx)
            n2_candidates = [j for j in neg_idx if j != n1]
            n2 = _R.choice(n2_candidates)
            return _hyb_sent(class_names[n1], class_names[n2])
        
    # ─────────────────────────── 1. No-Finding ──────────────────────────
    if labels[NF] == 1 or not pos_idx:
        pos_sent = _pos_sent(class_names[_R.choice(range(NF))])
        neg_sent = _neg_or_nofinding()
        hyb_sent = _hyb_sent(class_names[_R.choice(range(NF))], class_names[_R.choice(range(NF))])
        choices  = [pos_sent, neg_sent, hyb_sent]
        if shuffle:
            _R.shuffle(choices)
        return choices, choices.index(neg_sent)

    if no_hyb:
        ans_type = _R.choice(["POS", "NEG"])
    else:
        ans_type = _R.choice(["POS", "NEG", "HYB"])

    if ans_type == "POS":
        pos_sent = _pos_sent(class_names[_R.choice(pos_idx)])
        neg_sent = _neg_or_nofinding(disease=class_names[_R.choice(pos_idx)])
        hyb_sent = _make_hyb_false(class_names, pos_idx, neg_idx)
        answer  = pos_sent

    elif ans_type == "NEG":
        neg_cls_idx = _R.choice(neg_idx)

        if len(neg_idx) == 1:
            pos_cls_idx = neg_cls_idx
        else:
            if _R.random() < 0.75:
                pos_cls_idx = neg_cls_idx
            else:
                other_negs = [i for i in neg_idx if i != neg_cls_idx]
                pos_cls_idx = _R.choice(other_negs)

        pos_sent = _pos_sent(class_names[pos_cls_idx])
        neg_sent = _neg_sent(class_names[neg_cls_idx])
        hyb_sent = _make_hyb_false(class_names, pos_idx, neg_idx)
        answer  = neg_sent

    else:  # HYB
        pos_sent = _pos_sent(class_names[_R.choice(neg_idx)])
        neg_sent = _neg_sent(class_names[_R.choice(pos_idx)])
        hyb_sent = _hyb_sent(class_names[_R.choice(pos_idx)], class_names[_R.choice(neg_idx)])
        answer  = hyb_sent

    choices = [pos_sent, neg_sent, hyb_sent]
    if shuffle:
        _R.shuffle(choices)
    return choices, choices.index(answer)

def get_positive_mask_for_prompt(
    batch_labels: torch.Tensor,
    prompt: str,
    disease_names: List[str],
    no_finding_index: int
) -> torch.Tensor:
    B, C = batch_labels.shape
    assert C == len(disease_names), "Label dimension and disease_names length do not match."

    prompt_stripped = prompt.strip()

    if prompt_stripped.lower() == "there is no finding.":
        pos_mask = batch_labels[:, no_finding_index] == 1
        return pos_mask

    if prompt_stripped.startswith("There is ") and " but no " in prompt_stripped:
        body = prompt_stripped[len("There is "):].rstrip(".").strip()
        parts = body.split(" but no ")
        if len(parts) != 2:
            return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

        disease_name_pos = parts[0].strip()
        disease_name_neg = parts[1].strip()

        if (disease_name_pos not in disease_names) or (disease_name_neg not in disease_names):
            return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

        idx_pos = disease_names.index(disease_name_pos)
        idx_neg = disease_names.index(disease_name_neg)

        pos_mask = (batch_labels[:, idx_pos] == 1) & (batch_labels[:, idx_neg] == 0)
        return pos_mask

    if prompt_stripped.startswith("There is no "):
        disease_name = prompt_stripped[len("There is no "):].strip().rstrip(".")
        is_negation = True
    elif prompt_stripped.startswith("There is "):
        disease_name = prompt_stripped[len("There is "):].strip().rstrip(".")
        is_negation = False
    else:
        return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

    if disease_name not in disease_names:
        return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

    disease_idx = disease_names.index(disease_name)

    if not is_negation:
        pos_mask = batch_labels[:, disease_idx] == 1
    else:
        pos_mask = batch_labels[:, disease_idx] == 0

    return pos_mask

def sample_pos_neg_for_all_prompts(
    batch_labels: torch.Tensor,
    prompts: List[str],
    disease_names: List[str],
    no_finding_index: int,
    num_negatives: int = 3,
) -> Dict[str, Any]:
    B = batch_labels.shape[0]
    device = batch_labels.device

    usage = [0] * B

    out_dict: Dict[str, Any] = {}

    for prompt in prompts:
        prompt_stripped = prompt.strip()

        pos_mask = get_positive_mask_for_prompt(batch_labels, prompt, disease_names, no_finding_index)
        pos_mask = pos_mask.to(device=device)
        neg_mask = ~pos_mask

        pos_indices = pos_mask.nonzero(as_tuple=False).flatten().tolist()
        neg_indices = neg_mask.nonzero(as_tuple=False).flatten().tolist()

        if len(pos_indices) == 0:
            continue

        if len(neg_indices) < num_negatives:
            continue

        min_pos_usage = min(usage[i] for i in pos_indices)
        candidate_pos = [i for i in pos_indices if usage[i] == min_pos_usage]
        pos_idx = random.choice(candidate_pos)

        hard_neg_indices: List[int] = []
        is_mixed = False

        if prompt_stripped.startswith("There is ") and " but no " in prompt_stripped:
            body = prompt_stripped[len("There is "):].rstrip(".").strip()
            parts = body.split(" but no ")
            if len(parts) == 2:
                disease_A = parts[0].strip()
                disease_B = parts[1].strip()
                if disease_A in disease_names and disease_B in disease_names:
                    idx_A = disease_names.index(disease_A)
                    idx_B = disease_names.index(disease_B)

                    hard_mask = (batch_labels[:, idx_A] == 1) & (batch_labels[:, idx_B] == 1)
                    hard_mask = hard_mask & neg_mask.to(device=batch_labels.device)
                    hard_neg_indices = hard_mask.nonzero(as_tuple=False).flatten().tolist()
                    is_mixed = True

        if is_mixed and len(hard_neg_indices) == 0:
            continue

        sampled_negs: List[int] = []

        if is_mixed:
            hard_neg_sorted = sorted(
                hard_neg_indices,
                key=lambda i: (usage[i], random.random())
            )

            num_from_hard = min(num_negatives, len(hard_neg_sorted))
            selected_hards = hard_neg_sorted[:num_from_hard]
            sampled_negs.extend(selected_hards)

            remaining = num_negatives - num_from_hard
            if remaining > 0:
                hard_set = set(hard_neg_indices)
                easy_neg_indices = [i for i in neg_indices if i not in hard_set]

                easy_neg_sorted = sorted(
                    easy_neg_indices,
                    key=lambda i: (usage[i], random.random())
                )

                if len(easy_neg_sorted) < remaining:
                    sampled_negs.extend(easy_neg_sorted)
                else:
                    sampled_negs.extend(easy_neg_sorted[:remaining])
        else:
            neg_sorted = sorted(
                neg_indices,
                key=lambda i: (usage[i], random.random())
            )
            sampled_negs = neg_sorted[:num_negatives]

        if len(sampled_negs) < num_negatives:
            continue

        idx_list = [pos_idx] + sampled_negs
        random.shuffle(idx_list)
        ans_idx = idx_list.index(pos_idx)

        for i in idx_list:
            usage[i] += 1

        out_dict[prompt] = (idx_list, ans_idx)

    return out_dict

def extract_disease_indices_from_prompt(
    prompt: str,
    disease_names: List[str],
) -> Set[int]:
    prompt_stripped = prompt.strip()
    found: Set[int] = set()

    if prompt_stripped.startswith("There is ") and " but no " in prompt_stripped:
        body = prompt_stripped[len("There is "):].rstrip(".").strip()
        parts = body.split(" but no ")
        if len(parts) == 2:
            disease_A = parts[0].strip()
            disease_B = parts[1].strip()
            if disease_A in disease_names:
                found.add(disease_names.index(disease_A))
            if disease_B in disease_names:
                found.add(disease_names.index(disease_B))

    lowered_prompt = prompt_stripped.lower()
    for idx, name in enumerate(disease_names):
        if name.lower() in lowered_prompt:
            found.add(idx)

    return found

def select_prompts_max_disease_coverage(
    mcq_dict: Dict[str, Any],
    disease_names: List[str],
    B: int,
) -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = list(mcq_dict.items())
    Q = len(items)
    if B >= Q:
        return mcq_dict

    prompts: List[str] = [p for p, _ in items]

    diseases_per_prompt: List[Set[int]] = [
        extract_disease_indices_from_prompt(p, disease_names)
        for p in prompts
    ]

    selected_indices: List[int] = []
    covered_diseases: Set[int] = set()
    remaining_indices: Set[int] = set(range(Q))

    for _ in range(B):
        best_gain = -1
        best_candidates: List[int] = []

        for i in remaining_indices:
            new_coverage = diseases_per_prompt[i] - covered_diseases
            gain = len(new_coverage)

            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain:
                best_candidates.append(i)

        if best_gain <= 0:
            if not remaining_indices:
                break
            remaining_list = list(remaining_indices)
            remaining_list.sort(
                key=lambda i: len(diseases_per_prompt[i]),
                reverse=True,
            )
            chosen = remaining_list[0]
        else:
            chosen = random.choice(best_candidates)

        selected_indices.append(chosen)
        covered_diseases |= diseases_per_prompt[chosen]
        remaining_indices.remove(chosen)

        if not remaining_indices:
            break

    selected_prompts = {prompts[i]: items[i][1] for i in selected_indices}

    return selected_prompts


def build_t2i_mcq_batch(
    batch: Dict[str, Any],
    tokenizer,
    prompts: List[str],
    class_names: List[str],
    max_length: int,
    num_negatives: int = 2,
    no_hyb: bool = False
) -> Dict[str, Any]:
    
    imgs: torch.Tensor = batch["imgs"]
    B, C, H, W = imgs.shape
    
    if no_hyb :
        prompts = [p for p in prompts if " but no " not in p]
    
    mcq_dict = sample_pos_neg_for_all_prompts(
        batch_labels=batch["label"],
        prompts=prompts,
        disease_names=class_names,
        no_finding_index=14,
        num_negatives=num_negatives
    )
    
    mcq_dict = select_prompts_max_disease_coverage(
    mcq_dict=mcq_dict,
    disease_names=class_names,
    B=B)
    
    prompts: List[str] = []
    all_idx_lists: List[List[int]] = []
    all_ans_indices: List[int] = []

    for prompt, (idx_list, ans_idx) in mcq_dict.items():
        if len(idx_list) == 0:
            continue
        prompts.append(prompt)
        all_idx_lists.append(idx_list)
        all_ans_indices.append(ans_idx)

    if len(prompts) == 0:
        return batch
    
    Q = len(prompts)
    N = len(all_idx_lists[0])

    idx_tensor = torch.tensor(all_idx_lists, dtype=torch.long)

    flat_indices = idx_tensor.view(-1)
    flat_imgs = imgs[flat_indices]
    mcq_imgs = flat_imgs.view(Q, N, C, H, W)

    tok = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    token_type_ids = tok.get("token_type_ids", None)

    input_ids = input_ids.unsqueeze(1).expand(Q, N, -1)
    attention_mask = attention_mask.unsqueeze(1).expand(Q, N, -1)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.unsqueeze(1).expand(Q, N, -1)

    answers = torch.tensor(all_ans_indices, dtype=torch.long)

    batch_out = dict(batch)
    batch_out["imgs"] = mcq_imgs
    batch_out["caption_ids"] = input_ids
    batch_out["attention_mask"] = attention_mask
    batch_out["token_type_ids"] = token_type_ids
    batch_out["answer_idx"] = answers
    
    return batch_out

def _sp(s: str) -> str:
    return s.replace("_", " ")

def _pick_k(rng: random.Random, pool: List[int], k: int) -> List[int]:
    k = max(0, min(k, len(pool)))
    return rng.sample(pool, k)

def generate_prompt(
    labels: List[int],
    class_names: List[str],
    n_findings_range: Tuple[int, int] = (2, 3),
    no_finding_name: str = "No Finding",
    seed: Optional[int] = None,
    p_multi: float = 0.45,
) -> str:
    assert len(labels) == len(class_names), "labels/class_names length mismatch"

    if seed is None:
        sysrand = random.SystemRandom()
        seed = sysrand.getrandbits(64)
    rng = random.Random(seed)

    try:
        nf_idx = class_names.index(no_finding_name)
    except ValueError:
        nf_idx = len(class_names) - 1

    disease_idx = [i for i in range(len(class_names)) if i != nf_idx]
    pos_idx = [i for i in disease_idx if labels[i] == 1]
    neg_idx = [i for i in disease_idx if labels[i] == 0]
    has_nf = (0 <= nf_idx < len(labels)) and (labels[nf_idx] == 1)

    n_min, n_max = n_findings_range
    n_target = rng.randint(n_min, n_max)

    POS_SINGLE_POOL = [
        "{x} is present.",
        "{x} is noted.",
        "There is {x}.",
        "There is evidence of {x}.",
        "Findings are consistent with {x}.",
    ]
    NEG_SINGLE_POOL = [
        "No {x} is identified.",
        "No {x} is seen.",
        "{x} is not present.",
        "{x} is absent.",
        "No radiographic evidence of {x}.",
        "There is no evidence of {x}.",
    ]

    pos_single_tpls = POS_SINGLE_POOL[:]
    neg_single_tpls = NEG_SINGLE_POOL[:]
    rng.shuffle(pos_single_tpls)
    rng.shuffle(neg_single_tpls)

    def _choose_tpl(pool: List[str], backup: List[str]) -> str:
        nonlocal rng
        if not pool:
            pool.extend(backup)
            rng.shuffle(pool)
        return pool.pop()

    def _fmt_pos(name: str) -> str:
        tpl = _choose_tpl(pos_single_tpls, POS_SINGLE_POOL)
        return tpl.format(x=_sp(name))

    def _fmt_neg(name: str) -> str:
        tpl = _choose_tpl(neg_single_tpls, NEG_SINGLE_POOL)
        return tpl.format(x=_sp(name))

    used_sentences = set()
    used_groups = set()

    def _try_add_sentence(s: str, kind: str, name: str) -> bool:
        key_sent = s.strip().lower()
        key_group = (kind, frozenset({name.strip().lower()}))
        if key_sent in used_sentences or key_group in used_groups:
            return False
        used_sentences.add(key_sent)
        used_groups.add(key_group)
        sentences.append(s)
        return True

    sentences: List[str] = []

    if has_nf or len(pos_idx) == 0:
        base = "There is no finding."
        used_sentences.add(base.strip().lower())
        used_groups.add(("neg", frozenset({"__no_finding__"})))
        sentences.append(base)

        neg_pool_names = [_sp(class_names[i]) for i in neg_idx] if neg_idx else [_sp(class_names[i]) for i in disease_idx]
        rng.shuffle(neg_pool_names)

        need = max(n_target - 1, 0)
        attempts = 0
        while len(sentences) < 1 + need and attempts < 10 * max(1, need):
            attempts += 1
            if not neg_pool_names:
                break
            name = rng.choice(neg_pool_names)
            cand = _fmt_neg(name)
            _try_add_sentence(cand, "neg", name)

        rng.shuffle(sentences)
        return " ".join(sentences).replace(".  ", ". ").strip()

    pos_take_idx = pos_idx[:]
    rng.shuffle(pos_take_idx)
    pos_names = [_sp(class_names[i]) for i in pos_take_idx]

    if n_target < len(pos_names):
        n_target = len(pos_names)

    for name in pos_names:
        cand = _fmt_pos(name)
        _try_add_sentence(cand, "pos", name)

    remain_neg_idx = [i for i in neg_idx if i not in pos_take_idx]
    n_neg_need = max(n_target - len(sentences), 0)
    neg_take_idx = _pick_k(rng, remain_neg_idx, min(len(remain_neg_idx), max(n_neg_need, 2)))
    neg_names = [_sp(class_names[i]) for i in neg_take_idx]
    rng.shuffle(neg_names)

    attempts = 0
    while len(sentences) < n_target and attempts < 10 * max(1, n_neg_need):
        attempts += 1
        if not neg_names:
            break
        name = rng.choice(neg_names)
        cand = _fmt_neg(name)
        _try_add_sentence(cand, "neg", name)

    if len(sentences) > n_target:
        rng.shuffle(sentences)
        sentences = sentences[:n_target]

    rng.shuffle(sentences)
    return " ".join(sentences).replace(".  ", ". ").strip()