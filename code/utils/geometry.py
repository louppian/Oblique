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

def fit_frame(z, t, ni, nt):
    """Estimate fixed mean/noise planes from reference embeddings."""
    z, t, ni, nt = (unit(x) for x in (z, t, ni, nt))
    frame = {}
    for modality, clean, noise in [('image', z, ni), ('text', t, nt)]:
        r, b = unit(clean.mean(0)), unit(noise.mean(0))
        gamma = float(r @ b)
        if 1 - gamma * gamma < 1e-10:
            raise ValueError(f'{modality}: rhat/bhat need independent directions.')
        q = basis([r, b])
        dual = np.column_stack((r - gamma * b, b - gamma * r)) / (1 - gamma * gamma)
        frame[modality] = dict(r=r, b=b, q=q, dual=dual)
    return frame

def residual(x, q):
    return x - x @ q @ q.T

def components(x, f):
    """u = alpha*rhat + beta*bhat + v; v is orthogonal to the mean/noise plane."""
    x = np.asarray(x)
    coefficients = x @ f['dual']
    r = coefficients[..., 0, None] * f['r']
    b = coefficients[..., 1, None] * f['b']
    return dict(Rhat=r, Bhat=b, V=x-r-b)
