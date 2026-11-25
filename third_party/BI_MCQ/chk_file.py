from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

BASE = Path("/shared/home/mai/Taehun/Uncertainty")

PRETRAIN_WEIGHT = (
    BASE / "MICCAI_2025/CARZero/pretrain_model/CARZero_best_model.ckpt"
)
FINETUNED_WEIGHT = (
    BASE / "MICCAI_2025/CARZero/checkpoints/best_model.ckpt"
)
CONFIG_FILE = BASE / "BI_MCQ/configs/chest14_finetuning_Bi-MCQ.yaml"

M3AE_WEIGHT = (
    BASE / "MICCAI_2025/CARZero/pretrain_model/VITB-16-M3AE_last.ckpt"
)

IMAGE_ROOT = BASE / "data/NIH"
LABEL_SOURCE = IMAGE_ROOT / "Data_Entry_2017.csv"
TEST_IMAGE_LIST = IMAGE_ROOT / "NIH_test_data.csv"

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia", "No Finding",
]

errors = []
warnings = []


def check_file(name, path):
    if not path.is_file():
        errors.append(f"{name}: 파일 없음 — {path}")
        return False
    if path.stat().st_size == 0:
        errors.append(f"{name}: 빈 파일 — {path}")
        return False
    print(f"[OK] {name}: {path}")
    return True


def read_csv(path, required):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: 필수 열 누락 {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path.name}: 데이터가 없습니다.")
    if df[required].isna().any().any():
        raise ValueError(f"{path.name}: 필수 열에 결측값이 있습니다.")

    df["Image Index"] = df["Image Index"].astype(str).str.strip()
    if df["Image Index"].eq("").any():
        raise ValueError(f"{path.name}: 빈 이미지 이름이 있습니다.")
    if df["Image Index"].duplicated().any():
        raise ValueError(f"{path.name}: 중복된 Image Index가 있습니다.")
    return df


def check_text_model(cfg):
    """다운로드 없이 로컬 폴더 또는 Hugging Face 캐시 확인."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    model_ref = str(cfg.model.text.bert_type)
    local_path = Path(model_ref).expanduser()

    try:
        if local_path.is_dir():
            model_dir = local_path
        else:
            model_dir = Path(snapshot_download(
                repo_id=model_ref,
                local_files_only=True,
            ))

        AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

        weights = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        ]
        indexes = [
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin.index.json",
        ]

        if not any(p.is_file() and p.stat().st_size > 0 for p in weights):
            import json

            index = next((p for p in indexes if p.is_file()), None)
            if index is None:
                raise ValueError("텍스트 모델 가중치 파일이 없습니다.")

            metadata = json.loads(index.read_text())
            shards = set(metadata["weight_map"].values())
            if not shards:
                raise ValueError("가중치 shard 목록이 비어 있습니다.")
            for shard in shards:
                p = model_dir / shard
                if not p.is_file() or p.stat().st_size == 0:
                    raise ValueError(f"가중치 shard 누락: {shard}")

        print(f"[OK] 텍스트 모델·토크나이저: {model_dir}")

    except Exception as exc:
        errors.append(
            f"텍스트 모델 로컬 준비 미확인: {model_ref}\n"
            f"    {exc}\n"
            "    온라인 실행 시 다운로드될 수 있습니다."
        )


# 1. 모델 파일
for name, path in [
    ("CARZero 사전학습 가중치", PRETRAIN_WEIGHT),
    ("Bi-MCQ 추론 가중치", FINETUNED_WEIGHT),
    ("M3AE 가중치", M3AE_WEIGHT),
]:
    check_file(name, path)

if check_file("설정 파일", CONFIG_FILE):
    try:
        cfg = OmegaConf.load(CONFIG_FILE)
        check_text_model(cfg)
    except Exception as exc:
        errors.append(f"설정/텍스트 모델 검사 실패: {exc}")

# 2. 이미지 검색 — GitHub 코드와 동일한 패턴
image_map = {}
duplicates = set()

for path in IMAGE_ROOT.glob("images*/*/*.png"):
    if not path.is_file():
        continue
    if path.name in image_map:
        duplicates.add(path.name)
    image_map[path.name] = path

print(f"[INFO] 이미지 검색: {IMAGE_ROOT}/images*/*/*.png")
print(f"[INFO] 고유 이미지 파일명: {len(image_map)}개")

if not image_map:
    errors.append("이미지를 찾지 못했습니다.")
if duplicates:
    errors.append(
        f"중복 이미지 파일명 {len(duplicates)}개: "
        + ", ".join(sorted(duplicates)[:10])
    )

# 3. 원본에서 15개 라벨 생성 + 테스트 이미지 목록 매칭
source_ok = check_file("원본 라벨 CSV", LABEL_SOURCE)
test_ok = check_file("테스트 이미지 목록 CSV", TEST_IMAGE_LIST)

if source_ok and test_ok:
    try:
        source = read_csv(
            LABEL_SOURCE, ["Image Index", "Finding Labels"]
        )
        test = read_csv(TEST_IMAGE_LIST, ["Image Index"])

        # Pleural_Thickening → Pleural Thickening
        normalized = (
            source["Finding Labels"]
            .astype(str)
            .str.replace("_", " ", regex=False)
        )
        label_sets = normalized.str.split("|").map(
            lambda values: {v.strip() for v in values}
        )

        observed = set().union(*label_sets)
        unknown = observed - set(CLASSES)
        if unknown:
            errors.append(f"알 수 없는 원본 라벨: {sorted(unknown)}")

        absent = set(CLASSES) - observed
        if absent:
            errors.append(f"원본에 없는 라벨: {sorted(absent)}")

        conflicting = label_sets.map(
            lambda labels: "No Finding" in labels and len(labels) > 1
        )
        if conflicting.any():
            errors.append(
                f"No Finding과 질환이 함께 있는 행: {conflicting.sum()}개"
            )

        # 문자열 부분 일치 대신 분리한 라벨의 정확한 일치로 생성
        for name in CLASSES:
            source[name] = label_sets.map(
                lambda labels: int(name in labels)
            )

        print(f"[INFO] 원본 이미지: {len(source)}개")
        print(f"[INFO] 원본 고유 라벨: {len(observed)}개")

        # 테스트 CSV의 기존 9개 질환 열은 사용하지 않음
        selected = test[["Image Index"]].merge(
            source[["Image Index"] + CLASSES],
            on="Image Index",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        missing_labels = selected.loc[
            selected["_merge"] != "both", "Image Index"
        ].tolist()

        if missing_labels:
            errors.append(
                f"원본에 없는 테스트 이미지 {len(missing_labels)}개: "
                + ", ".join(missing_labels[:10])
            )
        else:
            print(f"[OK] 테스트 라벨 매칭: {len(selected)}개")
            print("[INFO] 테스트 라벨별 양성 개수:")
            print(selected[CLASSES].sum().astype(int).to_string())

        missing_images = [
            name for name in test["Image Index"]
            if name not in image_map
        ]
        if missing_images:
            errors.append(
                f"테스트 이미지 파일 누락 {len(missing_images)}개: "
                + ", ".join(missing_images[:10])
            )
        else:
            print(f"[OK] 테스트 이미지 파일 매칭: {len(test)}개")

    except Exception as exc:
        errors.append(f"라벨 검사 실패: {exc}")

warnings.extend([
    "체크포인트 내부 손상·Bi-MCQ 모델 호환성은 검사하지 않았습니다.",
    "테스트 이미지 목록이 공식 split과 같은지는 검사하지 않았습니다.",
    "이 스크립트의 경로 설정은 실제 추론 코드에 자동 반영되지 않습니다.",
    "두 CSV를 쓰려면 추론의 load_test_data()도 수정해야 합니다.",
])

for warning in warnings:
    print(f"[주의] {warning}")

if errors:
    print("\n[FAIL]")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("\n[PASS] 파일 준비·라벨 생성·테스트 이미지 매칭 검사 통과")