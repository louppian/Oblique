"""COCO CLIP: train+val -> two tables, three figures, decompostion.md."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache/clip/coco'


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
    for at, label in [(0, rf'$\hat{{\vec b}}_{sub}$'+'\n0.000°'),
                      (theta, rf'$\hat{{\vec r}}_{sub}$'+f'\n{theta:.3f}°')]:
        ax.text(at, 1.025, label, ha='center', transform=ax.get_xaxis_transform(), fontsize=11)


def style(ax):
    ax.grid(axis='y', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


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
        ax.text(.035, .87, f'평균 {angle.mean():.3f}°\n표준편차 {angle.std():.3f}°', transform=ax.transAxes)
        ax = axes[1, j]
        ax.hist(length, bins=np.linspace(0, 1, 51), density=True, color=color, alpha=.8, edgecolor='white', lw=.3)
        ax.axvline(length.mean(), color=color, ls='--')
        ax.set(xlim=(0, 1), xlabel='잔차크기(평면에서 멀어짐)', ylabel='밀도')
        ax.text(.035, .86, f'평균 {length.mean():.3f}\n표준편차 {length.std():.3f}', transform=ax.transAxes)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        for deg in [30, 45, 60]:
            ax.axvline(np.sin(np.deg2rad(deg)), color='#777', ls=':', lw=.8)
        secondary = ax.secondary_xaxis('top')
        secondary.set_xticks(np.sin(np.deg2rad([30, 45, 60])))
        secondary.set_xticklabels(['0.500\nθ=30°', '0.707\nθ=45°', '0.866\nθ=60°'], fontsize=10)
    for ax in axes.flat:
        style(ax)
    fig.subplots_adjust(left=.075, right=.97, top=.83, bottom=.10, hspace=.52, wspace=.20)
    fig.text(.5, .025, 'θ: 출력과 각자 평면 사이 각도 · 잔차크기 = sin θ · 표준편차는 전체 표본 수 기준', ha='center')
    fig.text(.075, .965, 'COCO train+val - OpenCLIP pretrain', fontsize=16, weight='bold')
    fig.savefig(OUT/'figure1.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True, sharey=True)
    for ax, (angle, length, _, stats), name, sub, color in zip(
            axes, data, ['이미지', '텍스트'], ['I', 'T'], ['#2767c4', '#d97724']):
        ax.scatter(angle, length, s=2 if len(angle)>1000 else 24,
                   alpha=.10 if len(angle)>1000 else .7, c=color, edgecolors='none', rasterized=True)
        references(ax, stats['theta'], sub, color)
        ax.set_title(f'{name} · {len(angle):,}개', pad=48, fontsize=16)
        ax.set(xlim=(-10, 130), ylim=(0, 1), xlabel='노이즈에서 멀어짐 (°)', ylabel='잔차크기(평면에서 멀어짐)')
        ax.text(.035, .08, f'x 표준편차 {angle.std():.3f}°\ny 표준편차 {length.std():.3f}', transform=ax.transAxes)
        ax.tick_params(labelleft=True)
        style(ax)
        for deg in [30, 45, 60]:
            ax.axhline(np.sin(np.deg2rad(deg)), color='#777', ls=':', lw=.8, alpha=.65)
    secondary = axes[1].secondary_yaxis('right')
    secondary.set_yticks(np.sin(np.deg2rad([30, 45, 60])))
    secondary.set_yticklabels(['θ=30°', 'θ=45°', 'θ=60°'])
    fig.subplots_adjust(left=.07, right=.91, top=.76, bottom=.17, wspace=.20)
    fig.text(.5, .035, '각자의 평면 기준 · 전체 표본 표시 · θ: 출력과 평면 사이 각도', ha='center')
    fig.text(.07, .96, 'COCO train+val - OpenCLIP pretrain', fontsize=16, weight='bold')
    fig.savefig(OUT/'figure2.png', dpi=170)
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
    lines = ['# COCO train+val CLIP 분해 결과', '',
        '## 모델과 자료 — COCO train+val - OpenCLIP pretrain', '',
        '- 이미지 인코더: **CLIP ViT-B/32**, 모델 설정 `ViT-B-32-quickgelu`.',
        '- 텍스트 인코더: **CLIP text Transformer**, 같은 CLIP 체크포인트의 텍스트 인코더.',
        '- 캐시의 `settings.encoder`와 `code/utils/encoders.py`의 `ClipEncoder`를 기준으로 기록했다.',
        '- 각 인코더의 512차원 출력을 단위벡터로 정규화했다.',
        '- COCO 2017 train+val: 이미지 123,287개(train 118,287 + val 5,000), 캡션 616,767개(train 591,753 + val 25,014). 두 분할을 합쳐 하나의 평균·노이즈 기준과 평면을 추정한다.',
        '- 이미지 노이즈: RGB 각 픽셀을 [0,1) 균등분포에서 생성한 224×224 이미지 1,000개. seed 0, SeedSequence([0,1101]).',
        '- 텍스트 노이즈: Random token 616,767개, seed 0. COCO 캡션 순서대로 각 캡션의 잘림 처리 후 본문 token 수와 같은 길이의 노이즈를 하나씩 생성하여 길이 분포를 정확히 맞췄다. SOT·EOT·padding을 유지하고 본문에서는 해당 ID를 제외했다. 출력별 L2 정규화 후 평균 방향을 추정한다. 캡션과 생성 노이즈의 평균 길이는 모두 11.545 token이다. 생성 token ID와 길이, 단위 출력을 `cache/clip/coco/trainval_random_token_noise.npz`에 저장한다.',
        '- 입력 캐시: `cache/clip/coco/trainval_image.npz`, `trainval_text.npz`, `trainval_random_token_noise.npz`, 이미지 노이즈는 기존 `noise.npz`.',
        '- 이미지와 텍스트는 각각 전체 참조 표본에서 자기 평균·노이즈 방향과 평면을 추정한다.',
        '- 표준편차의 분모는 전체 표본 수이며 수치는 소수점 셋째 자리까지 표시한다.',
        f"- 체크포인트 SHA-256: `{metadata['settings']['checkpoint']['sha256']}`.", '',
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
        values = [0 if abs(v)<.0005 else v for v in values]
        lines.append(f'| {meaning} | {notation} | {values[0]:.3f} | {values[1]:.3f} |')
    lines += ['', '전체 평균 방향을 추정한 표본과 잔차 집계 표본이 같으므로 평균 잔차는 수치 오차 수준에서 0이다. 따라서 가중 방향 집중도는 0, 가중 방향 분산은 1이다.', '',
              '## 표 2. 두 인코더 사이 방향 관계', '',
              '| 추정모수(의미) | 추정량 | 이미지 | 텍스트 | 각도 (°) |', '|---|---|---|---|---:|']
    for i, (left, a) in enumerate(zip(['평균', '노이즈', 'Imageness'], ['r', 'b', 'm'])):
        for j, (right, b) in enumerate(zip(['평균', '노이즈', 'Imageness'], ['r', 'b', 'm'])):
            lines.append(rf'| {left}–{right} | $\arccos(\hat{{\vec {a}}}_I^\top\hat{{\vec {b}}}_T)$ | $\hat{{\vec {a}}}_I$ | $\hat{{\vec {b}}}_T$ | {angles[i,j]:.3f} |')
    lines += ['', '두 평면의 비교에는 다음 행렬식 기반 코사인 정의를 사용한다.', '',
        r'$$\frac{|(\hat{\vec r}_I^\top\hat{\vec r}_T)(\hat{\vec b}_I^\top\hat{\vec b}_T)-(\hat{\vec r}_I^\top\hat{\vec b}_T)(\hat{\vec b}_I^\top\hat{\vec r}_T)|}{\sqrt{1-\hat\gamma_I^2}\sqrt{1-\hat\gamma_T^2}}.$$','',
        f'값은 **{plane:.3f}**, 이 정의에 대응하는 각도는 **{np.degrees(np.arccos(plane)):.3f}°**이다.', '',
        '## 그림 1. 2×2 히스토그램', '', '![그림 1](figure1.png)', '',
        '## 그림 2. 1×2 산점도', '', '![그림 2](figure2.png)', '',
        '## 그림 3. 두 평면과 단위초구 위 호', '', '![그림 3](figure3.png)', '',
        '각 인코더의 평균–노이즈 각도를 보존하고, 네 방향의 인코더 간 내적 오차를 최소화하여 3차원에 배치했다. 표 2는 원래 512차원의 측정값이며 그림 3은 시각화를 위한 근사이다.', '',
        f"인코더 간 코사인 근사의 RMSE: {metadata['figure3_cosine_rmse']:.3f}.", '',
        '## 분포의 의미', '',
        f"이미지의 노이즈에서 멀어짐과 잔차크기 상관은 {stats[0]['correlation']:.3f}, 텍스트는 {stats[1]['correlation']:.3f}이다. 두 인코더 모두 노이즈에서 멀어질수록 잔차크기가 커지는 경향이며, 텍스트에서 더 뚜렷하다.", '',
        '이미지는 평균·노이즈 두 방향 밖에 남는 개별 이미지의 성분을, 텍스트는 두 방향 밖에 남는 개별 캡션의 성분을 보여준다. 물체·장면·구도와 문장의 내용·표현 차이가 가능한 원인이며, 원인 구분은 해당 이미지와 캡션을 연결하여 확인한다.', '',
        '특히 텍스트에서 노이즈 방향에 가까운 캡션은 평균–노이즈 평면 밖 성분도 상대적으로 작다. 노이즈에서 멀어질수록 두 기준 방향으로 설명한 뒤 남는 캡션별 성분이 커진다. 이는 관측된 표본 간 관계이며, 노이즈점으로 수렴한다는 판단에는 노이즈에서 멀어짐과 잔차크기가 함께 0으로 작아지는 과정의 확인이 필요하다.', '',
        '같은 산점도 위치의 출력도 서로 다른 잔차 방향을 가질 수 있다. 잔차크기는 평균·노이즈 평면으로 설명한 뒤 남는 성분의 양을 나타낸다.', '',
        r'$$\operatorname{tr}(\hat\Sigma)=\frac1N\sum\|\vec v\|^2-\left\|\frac1N\sum\vec v\right\|^2.$$','',
        f"잔차크기의 표준편차는 이미지 {stats[0]['residual'][1]:.3f}, 텍스트 {stats[1]['residual'][1]:.3f}이며, 잔차 전체의 분산은 각각 {stats[0]['variance']:.3f}, {stats[1]['variance']:.3f}이다. 전자는 길이의 변동을, 후자는 잔차 벡터 전체의 퍼짐을 나타낸다. 텍스트는 잔차 길이의 변동이 더 크고, 이미지는 잔차 전체의 분산이 더 크다.", '',
        '## 재현', '',
        '`C:/Users/USER/anaconda3/envs/torch_ev/python.exe D:/Oblique/code/analysis/COCO/decomposition.py`', '',
        '기존 이미지·캡션 캐시를 읽고 Random token 출력을 메모리에서 생성하여 figure1.png, figure2.png, figure3.png, decompostion.md를 갱신한다. noise.npz의 이미지 노이즈를 사용하며 텍스트 노이즈는 이 코드에서 새로 생성한다.']
    path = OUT/'decompostion.md'
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    marker = '\n## 선행연구와의 연결\n'
    literature = marker + previous.split(marker, 1)[1] if marker in previous else ''
    path.write_text('\n'.join(lines)+'\n'+literature, encoding='utf-8')


def random_token_noise(texts, cached, metadata):
    import hashlib
    if (CACHE/'trainval_random_token_noise.npz').exists():
        with np.load(CACHE/'trainval_random_token_noise.npz') as z:
            assert z['unit'].shape == (len(texts), 512)
            return z['unit']
    import torch
    import open_clip
    torch.set_num_threads(8)
    checkpoint = next((ROOT/'ckpt').glob('**/open_clip_model.safetensors'))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata['settings']['checkpoint']['sha256']
    model = open_clip.create_model('ViT-B-32-quickgelu', pretrained=str(checkpoint)).eval().requires_grad_(False).to('cuda')
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    tokens = tokenizer(list(texts))
    lengths = tokens.argmax(-1).numpy()-1
    rng = np.random.default_rng(0)
    vocabulary = np.array(sorted(set(tokenizer.encoder.values())-{0, tokenizer.sot_token_id, tokenizer.eot_token_id}))
    random = np.zeros((len(texts), tokens.shape[1]), dtype=np.int64)
    random[:, 0] = tokenizer.sot_token_id
    for j, length in enumerate(lengths):
        random[j, 1:length+1] = rng.choice(vocabulary, int(length))
        random[j, length+1] = tokenizer.eot_token_id
    with torch.inference_mode():
        output = np.empty((len(texts), 512), np.float32)
        for start in range(0, len(texts), 2048):
            output[start:start+2048] = torch.cat([model.encode_text(t, normalize=True) for t in torch.from_numpy(random[start:start+2048]).to('cuda').split(256)]).float().cpu().numpy()
            if start % 40960 == 0: print(f'Noise {start+len(output[start:start+2048])}/{len(texts)}', flush=True)
        check = model.encode_text(tokens[:32].to('cuda'), normalize=True).float().cpu().numpy()
    assert np.max(abs(check-cached[:32])) < 1e-4
    np.savez_compressed(CACHE/'trainval_random_token_noise.npz', unit=output, tokens=random, lengths=lengths, metadata=json.dumps(dict(method='random_token', count=len(texts), seed=0, length_matching='one_per_caption_in_cache_order', checkpoint=metadata['settings']['checkpoint'])))
    print(f'Random token noise: {len(output)}, concentration={np.linalg.norm(unit(output).mean(0)):.6f}', flush=True)
    return output


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
    fig.text(.07,.94,'COCO train+val - OpenCLIP pretrain',fontsize=25,weight='bold')
    fig.text(.07,.885,'원점을 지나는 두 평면 · 파랑: 이미지 · 주황: 텍스트 · 굵은 곡선: 단위초구 위 호',fontsize=15)
    fig.text(.5,.075,f"이미지 호 {np.degrees(theta[0]):.3f}°   |   텍스트 호 {np.degrees(theta[1]):.3f}°   |   측정 기하의 3차원 근사",ha='center',fontsize=14)
    fig.savefig(OUT/'figure3.png',dpi=170);plt.close(fig)
    return float(np.sqrt(np.mean(fit.fun**2)))



def combined_inputs():
    import time
    import torch
    import open_clip
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    torch.set_num_threads(8)
    rows = [json.loads((ROOT/f'data/coco/annotations/captions_{split}2017.json').read_text()) for split in ['train','val']]
    captions = [r for a in rows for r in sorted(a['annotations'],key=lambda r:r['id'])]
    images = [(split,r) for split,a in zip(['train','val'],rows) for r in sorted(a['images'],key=lambda r:r['id'])]
    texts = [r['caption'].strip() for r in captions]
    assert len(images)==123287 and len(texts)==616767
    model = None
    def load_model():
        return open_clip.create_model_and_transforms('ViT-B-32-quickgelu',pretrained=str(next((ROOT/'ckpt').glob('**/open_clip_model.safetensors'))))
    if not (CACHE/'trainval_text.npz').exists():
        model,_,preprocess=load_model();model=model.eval().requires_grad_(False).to('cuda')
        tokenizer=open_clip.get_tokenizer('ViT-B-32-quickgelu')
        outputs=np.empty((len(texts),512),np.float32)
        with torch.inference_mode():
            for start in range(0,591753,2048):
                end=min(start+2048,591753)
                tokens=tokenizer(texts[start:end]).to('cuda')
                outputs[start:end]=torch.cat([model.encode_text(t,normalize=True) for t in tokens.split(256)]).float().cpu().numpy()
                if start%40960==0: print(f'Train captions {end}/591753',flush=True)
        with np.load(CACHE/'text.npz') as z: outputs[591753:]=z['unit']
        np.savez_compressed(CACHE/'trainval_text.npz',unit=outputs,texts=np.asarray(texts),ids=np.array([r['id'] for r in captions]),image_id=np.array([r['image_id'] for r in captions]))
    with np.load(CACHE/'trainval_text.npz') as z: text_unit=z['unit']
    with np.load(CACHE/'noise.npz') as z:metadata=json.loads(str(z['metadata']))
    if model is not None: del model
    text_noise=random_token_noise(texts,text_unit,metadata)
    if not (CACHE/'trainval_image.npz').exists():
        paths=[ROOT/f"data/coco/{split}2017/{r['file_name']}" for split,r in images]
        while sum(1 for _ in (ROOT/'data/coco/train2017').glob('*.jpg'))<118287:
            print('Waiting for train2017 image extraction',flush=True);time.sleep(30)
        assert all(p.is_file() for p in paths)
        model,_,preprocess=load_model();model=model.eval().requires_grad_(False).to('cuda')
        outputs=np.empty((len(paths),512),np.float32)
        def read(path):
            with Image.open(path) as image:return preprocess(image.convert('RGB'))
        with ThreadPoolExecutor(8) as pool,torch.inference_mode():
            for start in range(0,118287,256):
                end=min(start+256,118287)
                batch=torch.stack(list(pool.map(read,paths[start:end]))).to('cuda')
                outputs[start:end]=model.encode_image(batch,normalize=True).float().cpu().numpy()
                if start%5120==0:print(f'Train images {end}/118287',flush=True)
        with np.load(CACHE/'image.npz') as z:outputs[118287:]=z['unit']
        np.savez_compressed(CACHE/'trainval_image.npz',unit=outputs,ids=np.array([r['id'] for _,r in images]),paths=np.array([str(p.relative_to(ROOT)) for p in paths]))
    with np.load(CACHE/'trainval_image.npz') as z:image_unit=z['unit']
    return image_unit,text_unit,text_noise,metadata

def main():
    threadpool_limits(8)
    image_unit,text_unit,text_noise,metadata=combined_inputs()
    with np.load(CACHE/'noise.npz') as noise:
        data=[analyze(image_unit,noise['image_unit']),analyze(text_unit,text_noise)]
    assert all(np.isfinite(d[0]).all() and np.isfinite(d[1]).all() for d in data)
    assert all(d[0].min() >= -10 and d[0].max() <= 130 for d in data)
    figures(data)
    metadata['figure3_cosine_rmse'] = figure3(data)
    report(data, metadata)
    print(f'Saved {OUT}: figure1.png, figure2.png, figure3.png, decompostion.md')


if __name__ == '__main__':
    main()
