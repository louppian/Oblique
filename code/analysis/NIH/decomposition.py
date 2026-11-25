"""NIH CARZero: existing caches -> two tables, two figures, decompostion.md."""
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
CACHE = ROOT / 'cache/carzero/nih'


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
    v = u-alpha[:, None]*r-beta[:, None]*b
    length = np.linalg.norm(v, axis=1)
    angle = np.degrees(np.arctan2(y, x))
    mean_v = np.linalg.norm(v.mean(0))
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
    fig.text(.075, .965, 'NIH - CARZero pretrain', fontsize=16, weight='bold')
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
    fig.text(.07, .96, 'NIH - CARZero pretrain', fontsize=16, weight='bold')
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
    lines = ['# NIH CARZero 분해 결과', '',
        '## 모델과 자료 — NIH - CARZero pretrain', '',
        '- 이미지 인코더: **ViT-B/16**, 체크포인트 설정 `vit_b_16`, 구현 `ImageEncoder`.',
        '- 텍스트 인코더: **Laihaoran/BioClinicalMPBERT**, 구현 `BertEncoder`.',
        '- 두 이름은 `ckpt/best_model.ckpt`의 `hyper_parameters.model`에서 확인했다.',
        '- Fusion 전 각 인코더의 768차원 전역 출력을 단위벡터로 정규화했다.',
        '- NIH 전체 이미지 112,120개, No Finding 포함 15항목 × 10템플릿 × 긍정·부정 = 300문장.',
        '- 이미지 노이즈: 그레이스케일 가우시안 1,000개, 평균 0.500, 표준편차 0.200, [0,1] 제한, uint8 내림 후 RGB 동일 복제, seed 0.',
        '- 텍스트 노이즈: 기존 문자 500개 + 단어 500개.',
        '- 입력 캐시: `cache/carzero/nih/all_image.npz`, `text.npz`, `noise.npz`.',
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
        '각 인코더의 평균–노이즈 각도를 보존하고, 인코더 간 네 내적의 오차를 최소화하여 3차원에 배치했다. 표 2는 원래 768차원의 측정값이며 그림 3은 시각화를 위한 근사이다.', '',
        f"인코더 간 코사인 근사의 RMSE: {metadata['figure3_cosine_rmse']:.3f}.", '',
        '## 분포의 의미', '',
        '이미지는 평균 방향 주변에서 평면 안의 방향이 비슷하면서도 잔차크기가 다양하다. 평균·노이즈 두 방향으로 설명되는 배치와, 두 방향 밖에 남는 개별 이미지 성분을 구분해서 읽는다.', '',
        'X-ray에서 x값이 작아지면서 잔차가 커지는 경향은, 평면 안의 방향이 노이즈 쪽으로 돌아가는 동시에 평균–노이즈 평면 밖의 성분도 커진다는 뜻이다. 두 기준 방향으로 설명되는 부분이 약해지고 다른 방향의 성분이 커지는 변화가 함께 나타난다.', '',
        '평균 방향을 데이터가 공유하는 흉부 표현으로 읽으면, 그 공통 표현에서 벗어난 정도가 증가하는 것으로 해석할 수 있다. 병변·비정상 구조와 촬영 자세·시야·노출 등의 차이가 가능한 원인이다. 원인에 대한 이 해석은 가설이며, 개별 X-ray와 라벨·촬영 조건을 연결하여 확인한다.', '',
        '텍스트는 평면 안의 방향 차이와 평면 밖 성분 양쪽에서 문장별 차이가 나타난다. 이 산점도는 잔차의 크기를 보여주며, 같은 위치의 점도 서로 다른 잔차 방향을 가질 수 있다.', '',
        '노이즈점으로 실제 수렴하려면 평면 안에서 노이즈 방향으로 향하는 것과 잔차가 0으로 작아지는 것이 함께 필요하다.', '',
        r'$$\operatorname{tr}(\hat\Sigma)=\frac1N\sum\|\vec v\|^2-\left\|\frac1N\sum\vec v\right\|^2.$$','',
        '잔차크기의 표준편차는 길이의 변동을, 잔차 전체의 분산은 벡터의 퍼짐을 나타낸다. 텍스트는 이미지보다 잔차 길이의 표준편차가 작고, 평균 잔차 길이와 잔차 전체의 분산은 크다. 이는 큰 잔차들이 비교적 비슷한 길이로 분포하는 구조이다.', '',
        '## 재현', '',
        '`C:/Users/USER/anaconda3/envs/torch_ev/python.exe D:/Oblique/code/analysis/NIH/decomposition.py`', '',
        '기존 캐시를 읽어 이 폴더의 figure1.png, figure2.png, figure3.png, decompostion.md를 갱신한다.']
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
    fig.text(.07,.94,'NIH - CARZero pretrain',fontsize=25,weight='bold')
    fig.text(.07,.885,'원점을 지나는 두 평면 · 파랑: 이미지 · 주황: 텍스트 · 굵은 곡선: 단위초구 위 호',fontsize=15)
    fig.text(.5,.075,f"이미지 호 {np.degrees(theta[0]):.3f}°   |   텍스트 호 {np.degrees(theta[1]):.3f}°   |   측정 기하의 3차원 근사",ha='center',fontsize=14)
    fig.savefig(OUT/'figure3.png',dpi=170);plt.close(fig)
    return float(np.sqrt(np.mean(fit.fun**2)))


def main():
    threadpool_limits(8)
    with np.load(CACHE/'noise.npz') as noise, np.load(CACHE/'text.npz') as text:
        metadata = json.loads(str(noise['metadata']))
        assert metadata['settings']['noise_family'] == 'grayscale_gaussian'
        assert text['unit'].shape == (300, 768)
        assert all(((text['disease']==d)&(text['polarity']==s)).sum()==10 for d in range(15) for s in ['pos','neg'])
        with np.load(CACHE/'all_image.npz') as image:
            assert json.loads(str(image['metadata']))['checkpoint'] == metadata['settings']['checkpoint']
            data = [analyze(image['unit'], noise['image_unit']), analyze(text['unit'], noise['text_unit'])]
    figures(data)
    metadata['figure3_cosine_rmse'] = figure3(data)
    report(data, metadata)
    print(f'Saved {OUT}: figure1.png, figure2.png, figure3.png, decompostion.md')


if __name__ == '__main__':
    main()
