from __future__ import annotations

import numpy as np

EPS = 1e-06

def unit(x, allow_zero=False):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('임베딩에 비유한 값이 있습니다.')
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not allow_zero and np.any(norm <= EPS):
        raise ValueError('평균/임베딩 방향이 정의되지 않습니다.')
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > EPS)

def basis(columns):
    x = np.column_stack(columns)
    (u, s, _) = np.linalg.svd(x, full_matrices=False)
    if not len(s) or s[0] <= EPS:
        return np.empty((x.shape[0], 0))
    return u[:, s > EPS * s[0]]

def covariance(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(0)
    return centered.T @ centered / len(x)

def fit_frame(z, t, ni, nt):
    """Training-only estimates: rhat, bhat, leading outside-plane PCA axis."""
    (z, t, ni, nt) = (unit(x) for x in (z, t, ni, nt))
    frame = {}
    for (modality, clean, noise) in [('image', z, ni), ('text', t, nt)]:
        (mu, nu) = (clean.mean(0), noise.mean(0))
        (r, b) = (unit(mu), unit(nu))
        gamma = float(r @ b)
        if 1 - gamma * gamma < 1e-10:
            raise ValueError(f'{modality}: rhat/bhat need independent directions.')
        q = basis([r, b])
        if q.shape[1] != 2 or clean.shape[1] < 4:
            raise ValueError('The four-term definition needs a rank-2 plane and d >= 4.')
        outside = residual(clean, q)
        (values, vectors) = np.linalg.eigh(covariance(outside))
        if values[-1] <= 1e-12:
            raise ValueError(f'{modality}: outside-plane variance is zero.')
        e = unit(residual(vectors[:, -1], q))
        e *= 1 if e[np.argmax(np.abs(e))] >= 0 else -1
        dual = np.column_stack((r - gamma * b, b - gamma * r)) / (1 - gamma * gamma)
        ell = float(np.linalg.norm(r - b))
        frame[modality] = dict(mu=mu, r=r, b=b, e=e, q=q, q3=np.column_stack((q, e)), dual=dual, a=(r - b) / ell, ell=ell, R=float(np.linalg.norm(mu)), Rb=float(np.linalg.norm(nu)), e_eigenvalue=float(values[-1]), e_eigengap=float(values[-1] - values[-2]), e_variance_fraction=float(values[-1] / np.maximum(values, 0).sum()))
    if frame['image']['e'] @ frame['text']['e'] < 0:
        frame['text']['e'] *= -1
        frame['text']['q3'][:, -1] *= -1
    (i, f) = (frame['image'], frame['text'])
    frame['spaces'] = {'conformity': basis([i['r'], f['r']]), 'imageness': basis([i['a'], f['a']]), 'noise': basis([i['b'], f['b']]), 'image_cone': i['q'], 'text_cone': f['q'], 'joint_cone': basis([i['r'], i['b'], f['r'], f['b']]), 'image_e': i['e'][:, None], 'text_e': f['e'][:, None], 'joint_e': basis([i['e'], f['e']]), 'image_plane_e': i['q3'], 'text_plane_e': f['q3'], 'joint_plane_e': basis([i['r'], i['b'], i['e'], f['r'], f['b'], f['e']])}
    (values, vectors) = np.linalg.eigh((covariance(z) + covariance(t)) / 2)
    frame['pca'] = vectors[:, np.argsort(values)[::-1]]
    return frame

def residual(x, q):
    return x - x @ q @ q.T

def components(x, f):
    """u = alpha*rhat + beta*bhat + projection_on_e + delta (no centering)."""
    coefficients = np.asarray(x) @ f['dual']
    r = coefficients[..., 0, None] * f['r']
    b = coefficients[..., 1, None] * f['b']
    e = (np.asarray(x) @ f['e'])[..., None] * f['e']
    delta = np.asarray(x) - r - b - e
    return dict(Rhat=r, Bhat=b, E=e, Delta=delta)
