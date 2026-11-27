from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/utils'))
from modality_gap_reproduction import markdown_table

def main():
    root=ROOT/'code/analysis'
    names=['CIFAR10_VIT-L_14','CIFAR100_VIT-B_16','AOKVQA_VIT-B_16','AOKVQA_SigLIP_VIT-B_16','COCO_VIT-L_14_224','COCO_SigLIP_VIT-B_16']
    summary=[];rephrasing=[];retrieval=[]
    for name in names:
        out=root/name/'modality_gap';p=out/'metrics.csv'
        if not p.exists():continue
        f=pd.read_csv(p)
        coco=name.startswith('COCO');aok=name.startswith('AOK')
        if coco:
            selected=f[(f.sigma==.01)&(f.epsilon.isin([0.,2.,5.]))&(f.step==1.)]
            for r in selected.itertuples():retrieval.append(dict(model=name,threshold=r.epsilon,clean_accuracy_pct=100*r.clean_accuracy,noisy_accuracy_pct=100*r.noisy_accuracy,robustness_pct=100*r.robustness_exact_caption))
            protocol='COCO val2017 이미지 5,000장과 캡션 25,014개 전체를 후보로 사용했다. 검색 1위 캡션이 해당 영상의 캡션이면 정답으로 집계한다. 강건성은 이동 후 잡음 없는 예측과 잡음 추가 후 예측의 정확한 캡션 ID 일치율이다. 임계값은 중심화한 텍스트 행렬의 원래 특이값에 적용한다. 각 잡음 수준에서 100회, seed 2026, 이동값 사이에 동일한 잡음을 사용한다. 그림 1은 임계값에 따른 정확도·강건성, 그림 2는 중심화 임베딩과 gap의 코사인, 그림 3은 gap 변화 곡선이다.'
        else:
            if not aok:f=f[f.kind=='gaussian'].rename(columns={'parameter':'sigma'})
            for sigma in (0.,.0225):
                a=f[(f.sigma==sigma)&(f.step==0)].iloc[0];b=f[(f.sigma==sigma)&(f.step==1)].iloc[0]
                summary.append(dict(model=name,noise_std=sigma,accuracy_before_pct=100*a.accuracy,accuracy_after_pct=100*b.accuracy,robustness_before_pct=100*a.robustness,robustness_after_pct=100*b.robustness))
            protocol=('A-OKVQA validation 질문 1,145개, 질문당 보기 4개. 텍스트는 질문 + 공백 + 보기다. 질문별 영상 한 장과 원래 보기 4개로 gap 및 PCA를 계산한다. ' if aok else 'CIFAR 테스트 이미지 10,000장 전체. 텍스트는 a photo of a {class}. 형식이며 밑줄을 공백으로 바꾼다. CIFAR-100 배포 메타데이터의 cra 표기는 crab으로 보정했다. ')
            protocol+='정규화한 원래 임베딩에서 텍스트를 이동하고 제곱 유클리드 거리로 예측한다. 잡음은 텍스트에 독립 Gaussian으로 추가하며 표에 표준편차를 적었다. 100회, seed 2026, 이동값 사이에 동일한 잡음을 사용한다. 모든 이동값에서 잡음 없는 원래 예측이 유지되는지 검사했다.'
        report=f'# {name}: modality gap 재현\n\n{protocol}\n\n'
        report+='모델 가중치는 고정했다. CLIP은 OpenAI 224px 체크포인트, SigLIP은 timm ViT-B-16-SigLIP WebLI 체크포인트를 사용했다. 논문에 세부 pretrained 식별자가 생략되어 있어 이 선택을 재현 조건으로 명시한다.\n\n'
        report+='표의 accuracy와 robustness는 0~1 비율이다. step=0.0000은 원본, step=1.0000은 계산한 gap 방향 전체 이동이다.\n\n'+markdown_table(f[f.step.isin([0.,1.])])+'\n'
        if not coco and not aok:
            q=pd.read_csv(p);q=q[(q.kind=='quantization')&q.step.isin([0.,1.])]
            report+='\n## 양자화\n\n양쪽 임베딩을 이동 후 [−3, 3]의 균일 구간 32·64·128·256개로 양자화했다.\n\n'+markdown_table(q)+'\n'
        if aok:
            if (out/'rephrasing.csv').exists():
                q=pd.read_csv(out/'rephrasing.csv');a=q[q.step==0].iloc[0];b=q[q.step==1].iloc[0]
                rephrasing.append(dict(model=name,accuracy_before_pct=100*a.accuracy,accuracy_after_pct=100*b.accuracy))
                report+='\n## 문장 변경\n\n질문당 500회. 정답은 direct_answers에서, 오답 세 개는 train·validation의 choices와 direct_answers를 합친 고유 답변 목록에서 정답 별칭을 제외하고 중복 없이 선택했다. 답변 목록의 분할 범위와 질문별 PCA는 명시적인 구현 선택이다.\n\n'+markdown_table(q)+'\n'
            else:report+='\n문장 변경 평가는 진행 중이다.\n'
        report+='\n논문의 잡음 범례 σ²와 본문의 σ 표기에 차이가 있어, 여기서는 실제 입력한 표준편차로 표기한다. COCO 부록의 원래 특이값 임계값과 본문에 서술된 분산 백분율의 대응은 저자 코드로 추가 확인할 항목이다.\n\n출처: https://arxiv.org/html/2603.29080v2\n'
        (out/'report.md').write_text(report,encoding='utf-8')
    text='# Chowers et al. (CVPR 2026) 재현\n\nCIFAR-10·CIFAR-100·A-OKVQA 본문과 ImageNet·MS-COCO 부록을 대상으로 한다. 현재 완료한 계산과 데이터 접근이 필요한 항목을 구분해 기록했다.\n\n'
    text+='## 원문·모델·폴더\n\n원문: [Is the Modality Gap a Bug or a Feature? A Robustness Perspective](https://arxiv.org/html/2603.29080v2). 로컬 본문 PDF와 arXiv v2의 부록 19쪽을 대조했다. 논문은 CIFAR-10(ViT-L/14), CIFAR-100(ViT-B/16), A-OKVQA(ViT-B/16·SigLIP), ImageNet(ViT-B/16·ViT-L/14·SigLIP), MS-COCO(ViT-L/14·SigLIP)를 해당 강건성 실험에 사용한다.\n\n'
    text+='산출물: `code/analysis/<데이터셋>_<모델>/modality_gap/`의 `reproduce.py`, `report.md`, `metrics.csv`, `figure*.png`. 임베딩은 `cache/clip/`, 원본 데이터는 `data/`, 체크포인트는 `ckpt/`에 저장한다. A-OKVQA 문장 변경 실행 코드는 `code/utils/modality_gap_rephrasing.py`이고 결과는 같은 모델 폴더에 모인다.\n\n'
    text+='## 기존 형식의 3항 분해 산출물\n\n각 모델 폴더 최상위에 `decomposition.py`, `decompostion.md`, `figure1.png`, `figure2.png`, `figure3.png`를 생성했다. 기존 COCO와 같은 평균·입력 노이즈·잔차 분해, 추정량 표 2개, 히스토그램·산점도·두 평면 그림 3개이며 수치는 소수점 네 자리다.\n\n'
    for name in names:
        text+=f'- [{name} 3항 분해]({name}/decompostion.md)\n'
    text+='\nCOCO 224px 결과는 `COCO_VIT-L_14_224`, 기존 336px 결과는 `COCO_VIT-L_14`에 저장한다. 입력 노이즈 인코딩과 원본 임베딩은 cache/에 저장하며 Git 업로드에서 제외한다.\n\n'
    text+='## Gaussian 결과\n\n아래 수치는 저장된 소수점 네 자리 비율을 백분율로 표시했다. 원본과 전체 gap 이동을 비교한다.\n\n'+markdown_table(pd.DataFrame(summary))+'\n\n'
    text+='## A-OKVQA 문장 변경\n\n'+(markdown_table(pd.DataFrame(rephrasing)) if rephrasing else '계산 중')+'\n\n'
    text+='## MS-COCO 부록\n\n잡음 표준편차 0.0100. threshold=0.0000은 전체 텍스트 공간에 직교하는 방향이며, full rank에서는 이동량이 수치 오차 수준이다. threshold를 키우면 더 많은 방향으로 이동하며 정확도와 강건성의 교환 관계가 생긴다.\n\n'+markdown_table(pd.DataFrame(retrieval))+'\n\n'
    text+='## 결과 해석\n\nCIFAR와 A-OKVQA의 Gaussian 실험에서는 정확한 직교 방향으로 이동할 때 잡음 없는 예측을 유지하면서 잡음 강건성이 증가했다. COCO에서는 특이값 임계값을 높여 이동량을 늘리면 강건성 상승과 함께 잡음 없는 검색 정확도가 낮아지는 구간이 나타났다. A-OKVQA 문장 변경은 전체 이동보다 중간 이동에서 정확도가 더 높을 수 있으므로, 전체 이동 결과와 전체 곡선을 함께 기록했다. 모든 실험은 인코더 가중치를 고정한 후처리 평가다.\n\n'
    text+='## 재현 조건과 남은 항목\n\n- OpenAI CLIP ViT-B/16·ViT-L/14는 224px, SigLIP은 ViT-B/16 WebLI 224px을 사용했다. 기존 COCO ViT-L/14@336px 캐시와 구분해 새로 추출했다. 저자의 정확한 pretrained 식별자는 본문에 생략되어 있다.\n- 최초 임베딩만 단위 정규화한다. 이동 후 점수는 `image·text − ||text||² / 2`이며, 후보 사이에 공통인 영상 제곱 노름을 생략한 제곱 유클리드 거리 순위다.\n- 본문 그림의 잡음 표기를 표준편차로 해석했다. 부록의 σ² 범례를 문자 그대로 해석하면 다른 조건이 되므로 저자 코드 확인 항목으로 남겼다.\n- A-OKVQA는 질문별 PCA 및 train·validation 답변 목록을 구현 선택으로 명시했다.\n- ImageNet 원본은 `data/imagenet`에 다운로드하도록 준비했다. 공식 서버 연결 시간초과, 공식 Hugging Face 저장소는 기존 계정에 403을 반환했다. 계정 접근 동의 후 `code/utils/prepare_imagenet_validation.py`로 50,000장을 받고 모델별 실행 코드에 연결한다.\n- ImageNet Fig. 13·18 및 Fig. 15의 400개 원문 프롬프트 실험은 대기 상태다. ImageNet의 분산 5% 임계값 정의도 원문 그림과 추가 대조할 항목이다. Fig. 15의 정확한 프롬프트 목록은 원문에 제공되지 않았다.\n- 따라서 현재 결과는 명시한 체크포인트·분할·잡음 해석에서 실행한 재현 결과다. ImageNet까지의 전체 완료 여부는 위 대기 항목과 함께 판단한다.\n'
    (root/'modality_gap_reproduction.md').write_text(text,encoding='utf-8')
    print('Report updated',root/'modality_gap_reproduction.md',flush=True)

if __name__=='__main__':main()
