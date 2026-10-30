"""第二遍（消纳）的完全前瞻上限：线性规划对照。

开机方式与第一遍配对固定不变，在全年 8760 h 上统一优化第二遍新增的充放电，
最大化吸收的剩余弃电。约束与启发式第二遍相同：
- 充电只用剩余弃电：Σ_j c[j,t] ≤ 剩余弃电[t]
- 放电只替代最小出力线以上的火电：Σ_j d[j,t] ≤ 火电出力[t] − 最小出力线[t]
- 功率：c ≤ 充电功率 − 第一遍充电，d ≤ 放电功率 − 第一遍放电
- SOC：s[t] = ρ·s[t−1] + η_c·c[t] − d[t]/η_d，0 ≤ s ≤ 可用容量 − 第一遍抬高量，年初、年末 s = 0
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .model import SystemData
from .pass1 import Pass1
from .pass2 import Pass2


def _spaces(sd: SystemData, p1: Pass1):
    st = p1.st
    raw = sd.net_load + st.ch_total - st.dis_total
    cur = np.maximum(p1.min_line - raw, 0.0)
    dsp = np.maximum(raw - p1.min_line, 0.0)
    return cur, dsp


def lp_bound(sd: SystemData, p1: Pass1, objective: str = "absorb") -> dict:
    """objective："absorb" 最大化吸收弃电（充电量）；"displace" 最大化替代火电（放电量）。"""
    T, S = sd.T, sd.storages
    J = len(S)
    st = p1.st
    cur, dsp = _spaces(sd, p1)
    n = 3 * T                                   # 每种储能：c, d, s
    nv = J * n
    lb, ub = np.zeros(nv), np.zeros(nv)
    eq_rows, eq_cols, eq_vals = [], [], []
    idx = np.arange(T)
    for j, s in enumerate(S):
        o = j * n
        ub[o:o + T] = np.where(cur > 1e-6, np.maximum(s.p_ch - st.pch[j], 0), 0)
        ub[o + T:o + 2 * T] = np.where(dsp > 1e-6, np.maximum(s.power - st.pdis[j], 0), 0)
        ub[o + 2 * T:o + 3 * T] = np.maximum(s.usable - st.lift[j], 0)
        ub[o + 3 * T - 1] = 0.0                  # 年末回到 0
        r = j * T + idx                          # s[t] − ρ s[t−1] − η_c c[t] + d[t]/η_d = 0
        eq_rows += [r, r, r, r[1:]]
        eq_cols += [o + 2 * T + idx, o + idx, o + T + idx, o + 2 * T + idx[:-1]]
        eq_vals += [np.ones(T), np.full(T, -s.eta_c), np.full(T, 1 / s.eta_d), np.full(T - 1, -s.rho)]
    A_eq = sp.csr_matrix((np.concatenate(eq_vals), (np.concatenate(eq_rows), np.concatenate(eq_cols))),
                         shape=(J * T, nv))
    ub_rows = np.concatenate([np.tile(idx, J), T + np.tile(idx, J)])
    ub_cols = np.concatenate([np.concatenate([j * n + idx for j in range(J)]),
                              np.concatenate([j * n + T + idx for j in range(J)])])
    A_ub = sp.csr_matrix((np.ones(ub_rows.size), (ub_rows, ub_cols)), shape=(2 * T, nv))
    b_ub = np.concatenate([cur, dsp])
    c_obj = np.zeros(nv)
    for j in range(J):
        if objective == "absorb":
            c_obj[j * n:j * n + T] = -1.0             # 最大化吸收的弃电
        else:
            c_obj[j * n + T:j * n + 2 * T] = -1.0     # 最大化放电，即替代的火电
    t0 = time.time()
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(J * T),
                  bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x
    return {"目标值_MWh": -res.fun, "用时_s": time.time() - t0,
            "充电": np.array([x[j * n:j * n + T] for j in range(J)]),
            "放电": np.array([x[j * n + T:j * n + 2 * T] for j in range(J)]),
            "剩余弃电": cur, "放电空间": dsp}


def compare(sd: SystemData, objective: str = "absorb") -> dict:
    """同一第一遍结果上，比较启发式第二遍与 LP 上限。"""
    p1 = Pass1(sd).run()
    lp = lp_bound(sd, p1, objective)
    ch1 = p1.st.pch.sum(1).copy()
    dis1 = p1.st.pdis.sum(1).copy()
    Pass2(sd, p1.st, p1.min_line, p1.cap_line).run()
    heur_c = p1.st.pch.sum(1) - ch1
    heur_d = p1.st.pdis.sum(1) - dis1
    month = sd.month
    hc = np.zeros(sd.T)
    # 启发式第二遍逐时充电 = 总充电 − 第一遍充电（按小时）
    pairs_c = np.zeros(sd.T)
    for j, c, d, e, _, tag in p1.st.pairs:
        pairs_c[c] += e
    hc = p1.st.ch_total - pairs_c
    lpc = lp["充电"].sum(0)
    return {
        "剩余弃电_MWh": float(lp["剩余弃电"].sum()),
        "启发式吸收_MWh": float(heur_c.sum()),
        "LP上限吸收_MWh": float(lp["充电"].sum()),
        "启发式放电_MWh": float(heur_d.sum()),
        "LP放电_MWh": float(lp["放电"].sum()),
        "分储能": {s.name: {"启发式充电": float(heur_c[j]), "LP充电": float(lp["充电"][j].sum()),
                          "启发式放电": float(heur_d[j]), "LP放电": float(lp["放电"][j].sum())}
                for j, s in enumerate(sd.storages)},
        "分月_启发式吸收": [float(hc[month == m].sum()) for m in range(12)],
        "分月_LP吸收": [float(lpc[month == m].sum()) for m in range(12)],
        "LP用时_s": lp["用时_s"],
    }
