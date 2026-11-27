"""COCO CLIP: train+val -> two tables, three figures, decompostion.md."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'code/analysis'
TITLE = ''


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def analyze(u, noise):
    u, noise = unit(u), unit(noise)
    mu = u.mean(0)
    R = np.linalg.norm(mu)
    r, b = mu / R, unit(noise.mean(0))
    g = r @ b
    c, x = u @ r, u @ b
    y = (c - g*x) / np.sqrt(1-g*g)
    alpha, beta = (c-g*x)/(1-g*g), (x-g*c)/(1-g*g)
    length = np.empty(len(u))
    residual_sum = np.zeros(u.shape[1])
    for start in range(0, len(u), 8192):
        v = u[start:start+8192]-alpha[start:start+8192,None]*r-beta[start:start+8192,None]*b
        length[start:start+8192] = np.linalg.norm(v, axis=1)
        residual_sum += v.sum(0)
        assert max(np.max(abs(v@r)), np.max(abs(v@b))) < 1e-10
    angle = np.degrees(np.arctan2(y, x))
    mean_v = np.linalg.norm(residual_sum/len(u))
    m = unit(r-b)
    assert np.max(abs(alpha**2+beta**2+2*g*alpha*beta+length**2-1)) < 1e-10
    assert max(np.max(abs(v@r)), np.max(abs(v@b))) < 1e-10
    stats = dict(N=len(u), R=R, noise_R=np.linalg.norm(noise.mean(0)), gamma=g,
                 theta=np.degrees(np.arccos(g)), arc=np.arccos(g),
                 phi=np.degrees(np.arccos(np.clip(r@m, -1, 1))),
                 mean_residual=mean_v, kappa=mean_v/length.mean(),
                 V=1-(mean_v/length.mean())**2,
                 variance=np.mean(length**2)-mean_v**2,
                 correlation=np.corrcoef(angle, length)[0, 1])
    for name, values in [('alpha', alpha), ('beta', beta), ('c', c), ('noise_cos', x),
                         ('C', R*c), ('h', (1+alpha-beta)/2), ('residual', length), ('angle', angle)]:
        stats[name] = [values.mean(), values.std(ddof=0)]
    return angle, length, np.array([r, b, m]), stats


def references(ax, theta, sub, color):
    ax.axvspan(0, theta, color=color, alpha=.04)
    ax.axvline(0, color='#666', lw=1)
    ax.axvline(theta, color=color, ls='--', lw=1.5)
    for at, label in [(0, rf'$\hat{{\vec b}}_{sub}$'+'\n0.0000°'),
                      (theta, rf'$\hat{{\vec r}}_{sub}$'+f'\n{theta:.4f}°')]:
        ax.text(at, 1.025, label, ha='center', transform=ax.get_xaxis_transform(), fontsize=11)


def style(ax):
    ax.grid(axis='y', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))


def save_figure(fig, name):
    temporary=OUT/(name+'.tmp.png')
    fig.savefig(temporary,dpi=170)
    temporary.replace(OUT/name)


def figures(data):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for j, ((angle, length, _, stats), name, sub, color) in enumerate(zip(
            data, ['이미지', '텍스트'], ['I', 'T'], ['#2767c4', '#d97724'])):
        ax = axes[0, j]
        ax.hist(angle, bins=np.arange(-10, 132, 2), density=True, color=color, alpha=.8, edgecolor='white', lw=.3)
        references(ax, stats['theta'], sub, color)
        ax.set_title(f'{name} · {len(angle):,}개', pad=48, fontsize=16)
        ax.set(xlim=(-10, 130), xlabel='노이즈에서 멀어짐 (°)', ylabel='밀도 (1/°)')
        ax.text(.035, .87, f'평균 {angle.mean():.4f}°\n표준편차 {angle.std():.4f}°', transform=ax.transAxes)
        ax = axes[1, j]
        ax.hist(length, bins=np.linspace(0, 1, 51), density=True, color=color, alpha=.8, edgecolor='white', lw=.3)
        ax.axvline(length.mean(), color=color, ls='--')
        ax.set(xlim=(0, 1), xlabel='잔차크기(평면에서 멀어짐)', ylabel='밀도')
        ax.text(.035, .86, f'평균 {length.mean():.4f}\n표준편차 {length.std():.4f}', transform=ax.transAxes)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        for deg in [30, 45, 60]:
            ax.axvline(np.sin(np.deg2rad(deg)), color='#777', ls=':', lw=.8)
        secondary = ax.secondary_xaxis('top')
        secondary.set_xticks(np.sin(np.deg2rad([30, 45, 60])))
        secondary.set_xticklabels(['0.5000\nθ=30°', '0.7071\nθ=45°', '0.8660\nθ=60°'], fontsize=10)
    for ax in axes.flat:
        style(ax)
    fig.subplots_adjust(left=.075, right=.97, top=.83, bottom=.10, hspace=.52, wspace=.20)
    fig.text(.5, .025, 'θ: 출력과 각자 평면 사이 각도 · 잔차크기 = sin θ · 표준편차는 전체 표본 수 기준', ha='center')
    fig.text(.075, .965, TITLE, fontsize=16, weight='bold')
    save_figure(fig,'figure1.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True, sharey=True)
    for ax, (angle, length, _, stats), name, sub, color in zip(
            axes, data, ['이미지', '텍스트'], ['I', 'T'], ['#2767c4', '#d97724']):
        ax.scatter(angle, length, s=2 if len(angle)>1000 else 24,
                   alpha=.10 if len(angle)>1000 else .7, c=color, edgecolors='none', rasterized=True)
        references(ax, stats['theta'], sub, color)
        ax.set_title(f'{name} · {len(angle):,}개', pad=48, fontsize=16)
        ax.set(xlim=(-10, 130), ylim=(0, 1), xlabel='노이즈에서 멀어짐 (°)', ylabel='잔차크기(평면에서 멀어짐)')
        ax.text(.035, .08, f'x 표준편차 {angle.std():.4f}°\ny 표준편차 {length.std():.4f}', transform=ax.transAxes)
        ax.tick_params(labelleft=True)
        style(ax)
        for deg in [30, 45, 60]:
            ax.axhline(np.sin(np.deg2rad(deg)), color='#777', ls=':', lw=.8, alpha=.65)
    secondary = axes[1].secondary_yaxis('right')
    secondary.set_yticks(np.sin(np.deg2rad([30, 45, 60])))
    secondary.set_yticklabels(['θ=30°', 'θ=45°', 'θ=60°'])
    fig.subplots_adjust(left=.07, right=.91, top=.76, bottom=.17, wspace=.20)
    fig.text(.5, .035, '각자의 평면 기준 · 전체 표본 표시 · θ: 출력과 평면 사이 각도', ha='center')
    fig.text(.07, .96, TITLE, fontsize=16, weight='bold')
    save_figure(fig,'figure2.png')
    plt.close(fig)


def report(data, metadata):
    stats = [d[3] for d in data]
    angles = np.degrees(np.arccos(np.clip(data[0][2] @ data[1][2].T, -1, 1)))
    cross = data[0][2][:2] @ data[1][2][:2].T
    plane = abs(np.linalg.det(cross))/np.sqrt((1-stats[0]['gamma']**2)*(1-stats[1]['gamma']**2))
    rows = [
        ('데이터 집중도', r'$\hat R=\|\hat{\vec\mu}\|$', 'R', None),
        ('노이즈 집중도', r'$\|\hat{\bar{\vec u}}\|$', 'noise_R', None),
        ('평균–노이즈 방향 코사인', r'$\hat\gamma=\hat{\vec r}^\top\hat{\vec b}$', 'gamma', None),
        ('평균–노이즈 방향 각도 (°)', r'$\hat\theta$', 'theta', None),
        ('단위초구 위 호의 길이 (rad)', r'$\hat\theta$', 'arc', None),
        ('Conformity–imageness 각도 (°)', r'$\arccos(\hat{\vec r}^\top\hat{\vec m})$', 'phi', None),
        ('평균 방향 좌표의 평균', r'$\frac1N\sum\alpha$', 'alpha', 0),
        ('평균 방향 좌표의 표준편차', r'$\alpha$의 표준편차', 'alpha', 1),
        ('노이즈 방향 좌표의 평균', r'$\frac1N\sum\beta$', 'beta', 0),
        ('노이즈 방향 좌표의 표준편차', r'$\beta$의 표준편차', 'beta', 1),
        ('평균 방향 정렬의 평균', r'$\frac1N\sum\vec u^\top\hat{\vec r}$', 'c', 0),
        ('노이즈 방향 코사인의 평균', r'$\frac1N\sum\vec u^\top\hat{\vec b}$', 'noise_cos', 0),
        ('Conformity 평균', r'$\frac1N\sum C$', 'C', 0),
        ('Imageness 평균', r'$\frac1N\sum h$', 'h', 0),
        ('Imageness 표준편차', r'$h$의 표준편차', 'h', 1),
        ('개별 잔차크기의 평균', r'$\hat L(\vec v)$', 'residual', 0),
        ('개별 잔차크기의 표준편차', r'$\|\vec v\|$의 표준편차', 'residual', 1),
        ('평균 잔차의 크기', r'$\hat R(\vec v)$', 'mean_residual', None),
        ('잔차 크기로 가중한 방향 집중도', r'$\hat\kappa$', 'kappa', None),
        ('잔차 크기로 가중한 방향 분산', r'$\hat V$', 'V', None),
        ('잔차 전체의 분산', r'$\operatorname{tr}(\hat\Sigma)$', 'variance', None),
        ('노이즈에서 멀어짐 평균 (°)', r'$s$의 평균', 'angle', 0),
        ('노이즈에서 멀어짐 표준편차 (°)', r'$s$의 표준편차', 'angle', 1),
        ('노이즈에서 멀어짐과 잔차크기 상관', 'Pearson 상관계수', 'correlation', None)]
    lines = ['# '+TITLE+' 분해 결과', '', '## 모델과 자료', '', *metadata['description'], '',
        '## 분해와 그림의 좌표', '',
        r'$$\vec u=\alpha\hat{\vec r}+\beta\hat{\vec b}+\vec v,\qquad\vec v^\top\hat{\vec r}=\vec v^\top\hat{\vec b}=0.$$','',
        r'$$\alpha=\frac{\vec u^\top\hat{\vec r}-\hat\gamma\vec u^\top\hat{\vec b}}{1-\hat\gamma^2},\qquad\beta=\frac{\vec u^\top\hat{\vec b}-\hat\gamma\vec u^\top\hat{\vec r}}{1-\hat\gamma^2}.$$','',
        '고정한 추정 기준에 조건부로 개별 좌표와 잔차를 계산한다. 아래첨자는 이미지 I, 텍스트 T만 사용한다.', '',
        r'- x축 **노이즈에서 멀어짐 (°)**: 평면 안의 방향 각도 $s$.',
        r'$$s=\operatorname{atan2}\!\left(\frac{\vec u^\top\hat{\vec r}-\hat\gamma\vec u^\top\hat{\vec b}}{\sqrt{1-\hat\gamma^2}},\ \vec u^\top\hat{\vec b}\right).$$','',
        'atan2 결과를 도로 변환한다. 노이즈 방향이 0°, 평균 방향이 세로 점선이다. 평균 방향을 넘어간 위치도 유지한다.',
        r'- y축 **잔차크기(평면에서 멀어짐)**: $\|\vec v\|=\|\vec u-\hat Q\vec u\|$.',
        r'- 그림의 보조 눈금 $\theta$는 출력과 자기 평면 사이 각도로, $\|\vec v\|=\sin\theta$이다. $\hat\theta$는 두 추정 방향 사이 각도이다.', '',
        '## 표 1. 인코더별 추정량', '', '| 추정모수(의미) | 추정량 | 이미지 | 텍스트 |', '|---|---|---:|---:|']
    for meaning, notation, key, index in rows:
        values = [s[key] if index is None else s[key][index] for s in stats]
        values = [0 if abs(v)<.00005 else v for v in values]
        lines.append(f'| {meaning} | {notation} | {values[0]:.4f} | {values[1]:.4f} |')
    lines += ['', '전체 평균 방향을 추정한 표본과 잔차 집계 표본이 같으므로 평균 잔차는 수치 오차 수준에서 0이다. 따라서 가중 방향 집중도는 0, 가중 방향 분산은 1이다.', '',
              '## 표 2. 두 인코더 사이 방향 관계', '',
              '| 추정모수(의미) | 추정량 | 이미지 | 텍스트 | 각도 (°) |', '|---|---|---|---|---:|']
    for i, (left, a) in enumerate(zip(['평균', '노이즈', 'Imageness'], ['r', 'b', 'm'])):
        for j, (right, b) in enumerate(zip(['평균', '노이즈', 'Imageness'], ['r', 'b', 'm'])):
            lines.append(rf'| {left}–{right} | $\arccos(\hat{{\vec {a}}}_I^\top\hat{{\vec {b}}}_T)$ | $\hat{{\vec {a}}}_I$ | $\hat{{\vec {b}}}_T$ | {angles[i,j]:.4f} |')
    lines += ['', '두 평면의 비교에는 다음 행렬식 기반 코사인 정의를 사용한다.', '',
        r'$$\frac{|(\hat{\vec r}_I^\top\hat{\vec r}_T)(\hat{\vec b}_I^\top\hat{\vec b}_T)-(\hat{\vec r}_I^\top\hat{\vec b}_T)(\hat{\vec b}_I^\top\hat{\vec r}_T)|}{\sqrt{1-\hat\gamma_I^2}\sqrt{1-\hat\gamma_T^2}}.$$','',
        f'값은 **{plane:.4f}**, 이 정의에 대응하는 각도는 **{np.degrees(np.arccos(plane)):.4f}°**이다.', '',
        '## 그림 1. 2×2 히스토그램', '', '![그림 1](figure1.png)', '',
        '## 그림 2. 1×2 산점도', '', '![그림 2](figure2.png)', '',
        '## 그림 3. 두 평면과 단위초구 위 호', '', '![그림 3](figure3.png)', '',
        '각 인코더의 평균–노이즈 각도를 보존하고, 네 방향의 인코더 간 내적 오차를 최소화하여 3차원에 배치했다. 표 2는 원래 임베딩 차원의 측정값이며 그림 3은 시각화를 위한 근사이다.', '',
        f"인코더 간 코사인 근사의 RMSE: {metadata['figure3_cosine_rmse']:.4f}.", '',
        '## 분포의 의미', '',
        f"이미지의 노이즈에서 멀어짐과 잔차크기 상관은 {stats[0]['correlation']:.4f}, 텍스트는 {stats[1]['correlation']:.4f}이다.", '',
        '같은 산점도 위치의 출력도 서로 다른 잔차 방향을 가질 수 있다. 잔차크기는 평균·노이즈 평면으로 설명한 뒤 남는 성분의 양을 나타낸다.', '',
        r'$$\operatorname{tr}(\hat\Sigma)=\frac1N\sum\|\vec v\|^2-\left\|\frac1N\sum\vec v\right\|^2.$$','',
        f"잔차크기의 표준편차는 이미지 {stats[0]['residual'][1]:.4f}, 텍스트 {stats[1]['residual'][1]:.4f}이며, 잔차 전체의 분산은 각각 {stats[0]['variance']:.4f}, {stats[1]['variance']:.4f}이다. 전자는 길이의 변동을, 후자는 잔차 벡터 전체의 퍼짐을 나타낸다.", '',
        '## 재현', '',
        f'`python {OUT / "decomposition.py"}`', '', '같은 데이터의 평균·입력 노이즈 방향으로 3항을 분해하고 표 2개와 그림 3개를 생성한다. 입력 노이즈 캐시는 cache/ 아래에 저장한다.']
    path = OUT/'decompostion.md'
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    marker = '\n## 선행연구와의 연결\n'
    literature = marker + previous.split(marker, 1)[1] if marker in previous else ''
    path.write_text('\n'.join(lines)+'\n'+literature, encoding='utf-8')


def figure3(data):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    theta = [np.deg2rad(d[3]['theta']) for d in data]
    # Preserve each measured arc exactly; fit four cross-encoder dot products in 3D.
    first = np.array([[1., 0, 0], [np.cos(theta[0]), np.sin(theta[0]), 0]])
    second = np.array([[1., 0, 0], [np.cos(theta[1]), np.sin(theta[1]), 0]])
    target = data[0][2][[1,0]] @ data[1][2][[1,0]].T
    def error(rot):
        return (first @ (second @ Rotation.from_rotvec(rot).as_matrix().T).T-target).ravel()
    fits = [least_squares(error, start) for start in [[.6,.9,.4], [2.,1.,-1.], [-1.,2.,1.]]]
    fit = min(fits, key=lambda f: np.sum(f.fun**2))
    second = second @ Rotation.from_rotvec(fit.x).as_matrix().T
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_axes([.04,.12,.92,.76], projection='3d', computed_zorder=False)
    for pair, angle, color, sub in zip([first,second],theta,['#1464de','#f07812'],['I','T']):
        b,r = pair
        e = (r-np.cos(angle)*b)/np.sin(angle)
        corners = np.array([a*b+c*e for a,c in [(-1.2,-1.2),(1.2,-1.2),(1.2,1.2),(-1.2,1.2)]])
        ax.add_collection3d(Poly3DCollection([corners],facecolors=color,edgecolors=color,alpha=.16,linewidths=2,zorder=1))
        for t in [-.6,0,.6]:
            line=np.array([-1.2*b+t*e,1.2*b+t*e]);ax.plot(*line.T,color=color,alpha=.2,lw=.7,zorder=2)
        t=np.linspace(0,2*np.pi,300);circle=np.cos(t)[:,None]*b+np.sin(t)[:,None]*e
        ax.plot(*circle.T,color=color,alpha=.65,lw=1.5,zorder=3)
        t=np.linspace(0,angle,100);arc=np.cos(t)[:,None]*b+np.sin(t)[:,None]*e
        ax.plot(*arc.T,color=color,lw=7,zorder=5)
        for v,label in [(b,'b'),(r,'r')]:
            ax.quiver(0,0,0,*v,color=color,linewidth=2,arrow_length_ratio=.07,zorder=5)
            ax.scatter(*v,color=color,s=45,edgecolors='white',zorder=6)
            ax.text(*(1.15*v),rf'$\hat{{\vec {label}}}_{sub}$',color=color,fontsize=24,zorder=7)
    ax.scatter(0,0,0,color='#444',s=20,zorder=6)
    ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),zlim=(-1.4,1.4))
    ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=32,azim=45)
    fig.text(.07,.94,TITLE,fontsize=25,weight='bold')
    fig.text(.07,.885,'원점을 지나는 두 평면 · 파랑: 이미지 · 주황: 텍스트 · 굵은 곡선: 단위초구 위 호',fontsize=15)
    fig.text(.5,.075,f"이미지 호 {np.degrees(theta[0]):.4f}°   |   텍스트 호 {np.degrees(theta[1]):.4f}°   |   측정 기하의 3차원 근사",ha='center',fontsize=14)
    save_figure(fig,'figure3.png');plt.close(fig)
    return float(np.sqrt(np.mean(fit.fun**2)))





CONFIG = {
 'CIFAR10_VIT-L_14':('cifar10_vitl14','ViT-L-14','CIFAR-10 test'),
 'CIFAR100_VIT-B_16':('cifar100_vitb16','ViT-B-16','CIFAR-100 test'),
 'AOKVQA_VIT-B_16':('aokvqa_vitb16','ViT-B-16','A-OKVQA validation'),
 'AOKVQA_SigLIP_VIT-B_16':('aokvqa_siglip_vitb16','ViT-B-16-SigLIP','A-OKVQA validation'),
 'COCO_VIT-L_14_224':('coco_val_vitl14_224','ViT-L-14','COCO val2017'),
 'COCO_SigLIP_VIT-B_16':('coco_val_siglip_vitb16','ViT-B-16-SigLIP','COCO val2017'),
}

def run(name):
    import torch,open_clip,hashlib
    global OUT,TITLE
    threadpool_limits(8);torch.set_num_threads(8)
    cache_name,model_name,split=CONFIG[name];cache=ROOT/'cache/clip'/cache_name
    OUT=ROOT/'code/analysis'/name;OUT.mkdir(parents=True,exist_ok=True)
    TITLE=f'{split} - {model_name} 224px'
    arrays=np.load(cache/'embeddings.npz');images=arrays['images'];texts=arrays['texts'].reshape(-1,arrays['texts'].shape[-1])
    if name.startswith('CIFAR'):
        import pyarrow.parquet as pq
        dataset='cifar100' if name.startswith('CIFAR100') else 'cifar10'
        table=pq.read_schema(ROOT/'data/cifar'/f'{dataset}_test.parquet')
        features=json.loads(table.metadata[b'huggingface'])['info']['features']
        classes=features['fine_label' if dataset=='cifar100' else 'label']['names']
        prompts=[f'a photo of a {("crab" if c=="cra" else c).replace("_"," ")}.' for c in classes]
    elif name.startswith('AOK'):
        rows=json.loads((ROOT/'data/aokvqa/aokvqa_v1p0_val.json').read_text())
        prompts=[r['question']+' '+c for r in rows for c in r['choices']]
    else:
        annotation=json.loads((ROOT/'data/coco/annotations/captions_val2017.json').read_text())
        rows=sorted(annotation['annotations'],key=lambda r:r['id'])
        assert np.array_equal(arrays['caption_ids'],[r['id'] for r in rows])
        prompts=[r['caption'] for r in rows]
    assert len(prompts)==len(texts)
    siglip='SigLIP' in model_name
    weight=ROOT/'ckpt'/('ViT-B-16-SigLIP/open_clip_pytorch_model.bin' if siglip else f'{model_name}.pt')
    noise_cache=ROOT/'cache/clip'/('decomposition_noise_'+model_name.lower())
    noise_cache.mkdir(parents=True,exist_ok=True)
    image_path=noise_cache/'image_noise.npy';text_path=cache/'random_token_noise.npy'
    if not image_path.exists() or not text_path.exists():
        if siglip:
            from open_clip.tokenizer import HFTokenizer
            model,_,_=open_clip.create_model_and_transforms(model_name,pretrained=str(weight),device='cuda')
            tokenizer=HFTokenizer(str(weight.parent),context_length=64,clean='canonicalize')
            specials=tokenizer.tokenizer.all_special_ids;vocabulary=np.setdiff1d(np.arange(len(tokenizer.tokenizer)),specials)
        else:
            model,_,_=open_clip.create_model_and_transforms(model_name,pretrained=str(weight),device='cuda',force_quick_gelu=True,load_weights_only=False)
            tokenizer=open_clip.get_tokenizer(model_name);specials=[0,49406,49407];vocabulary=np.arange(1,49406)
        model.eval().requires_grad_(False)
        with torch.inference_mode(),torch.autocast('cuda'):
            if not image_path.exists():
                rng=np.random.default_rng(np.random.SeedSequence([0,1101]));encoded=[]
                mean=torch.tensor([.5]*3 if siglip else [.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
                std=torch.tensor([.5]*3 if siglip else [.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
                for start in range(0,1000,32):
                    pixels=rng.uniform(0,1,(min(32,1000-start),224,224,3)).astype(np.float32)
                    batch=torch.from_numpy(pixels).permute(0,3,1,2).cuda()
                    encoded.append(model.encode_image((batch-mean)/std,normalize=True).float().cpu().numpy())
                np.save(image_path,np.concatenate(encoded))
            if not text_path.exists():
                tokens=tokenizer(prompts).numpy();mask=~np.isin(tokens,specials)
                rng=np.random.default_rng(0);random=tokens.copy();random[mask]=rng.choice(vocabulary,int(mask.sum()))
                np.save(cache/'random_tokens.npy',random);np.save(cache/'token_lengths.npy',mask.sum(1))
                encoded=[]
                for start in range(0,len(random),256):encoded.append(model.encode_text(torch.from_numpy(random[start:start+256]).cuda(),normalize=True).float().cpu().numpy())
                np.save(text_path,np.concatenate(encoded))
        del model;torch.cuda.empty_cache()
    image_noise=np.load(image_path);text_noise=np.load(text_path);lengths=np.load(cache/'token_lengths.npy')
    assert image_noise.shape==(1000,images.shape[-1]) and text_noise.shape==texts.shape
    data=[analyze(images,image_noise),analyze(texts,text_noise)]
    assert all(np.isfinite(d[0]).all() and np.isfinite(d[1]).all() for d in data)
    digest=hashlib.sha256()
    with weight.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    metadata={'description':[
        f'- 모델: {model_name}, '+('WebLI SigLIP' if siglip else 'OpenAI CLIP')+f', 224px, 출력 {images.shape[-1]}차원.',
        f'- 표본: {split}, 이미지 출력 {len(images):,}개, 텍스트 출력 {len(texts):,}개. 각 출력을 단위 정규화한다.',
        '- 이미지 노이즈: 224×224 RGB 균등분포 [0,1) 1,000개, SeedSequence([0,1101]). 모델의 이미지 정규화를 적용한다.',
        f'- 텍스트 노이즈: 본문 token 수를 각각 맞춘 random token {len(texts):,}개, seed 0. 특수 token 위치와 padding을 유지하고 특수 ID를 제외한 어휘에서 추출한다. 평균 본문 길이 {lengths.mean():.4f}.',
        '- CIFAR는 클래스 프롬프트 각각 한 개, A-OKVQA는 질문–보기 4개씩, COCO는 전체 캡션을 사용한다. A-OKVQA 이미지 출력은 질문별 표본이므로 같은 영상이 반복될 수 있다.',
        '- 각 모달리티의 전체 표본으로 평균 방향을, 같은 인코더의 입력 노이즈 출력으로 노이즈 방향을 구한다. 기존 COCO와 같은 비직교 3항 분해를 적용한다.',
        '- 소수점 네 자리, 표준편차는 전체 표본 수 기준. 그림은 전체 표본을 표시한다.',
        f'- 데이터 캐시: `{cache.relative_to(ROOT)}/embeddings.npz`. 이미지 노이즈: `{image_path.relative_to(ROOT)}`. 텍스트 노이즈: `{text_path.relative_to(ROOT)}`.',
        f'- 체크포인트 SHA-256: `{digest.hexdigest()}`.',
    ]}
    figures(data);metadata['figure3_cosine_rmse']=figure3(data);report(data,metadata)
    (cache/'decomposition_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete',name,[(d[3]['N'],float(d[0].min()),float(d[0].max())) for d in data],flush=True)

if __name__=='__main__':
    for name in CONFIG:run(name)
