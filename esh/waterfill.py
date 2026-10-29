"""灌水法：x_i = clip(v_i − λ, 0, cap_i)，求水位 λ 使 Σ w_i·x_i = target。

- 充电：v 为弃电深度，最深的谷先削到和第二深的一样，再一起往下削；
- 放电：v 为净负荷/火电出力，最大的峰先削到和第二大的一样，再一起往下削。

Σ w·x 关于 λ 分段线性且单调不增，在断点上二分后线性插值得到精确解。
"""
from __future__ import annotations

import numpy as np


def waterfill(values, caps, weights, target: float, floor: float | None = None) -> np.ndarray:
    v = np.asarray(values, float)
    cap = np.maximum(np.asarray(caps, float), 0.0)
    w = np.asarray(weights, float)
    if v.size == 0 or target <= 0:
        return np.zeros_like(v)

    def f(lam: float) -> float:
        return float(np.dot(w, np.clip(v - lam, 0.0, cap)))

    lam_lo = float((v - cap).min())          # 此水位以下全部达到上限
    if floor is not None:
        lam_lo = max(lam_lo, floor)
    if f(lam_lo) <= target:
        return np.clip(v - lam_lo, 0.0, cap)

    bps = np.unique(np.concatenate([v, v - cap]))
    bps = np.concatenate([[lam_lo], bps[bps > lam_lo]])
    lo, hi = 0, len(bps) - 1                 # f(bps[lo]) > target ≥ f(bps[hi]) = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(bps[mid]) > target:
            lo = mid
        else:
            hi = mid
    f_lo, f_hi = f(bps[lo]), f(bps[hi])
    lam = bps[lo] + (f_lo - target) / (f_lo - f_hi) * (bps[hi] - bps[lo])
    return np.clip(v - lam, 0.0, cap)
