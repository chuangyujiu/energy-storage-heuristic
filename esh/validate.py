"""校验：用最终充放电结果反算逐时火电出力并检查各项约束。"""
from __future__ import annotations

import numpy as np

from .model import SystemData
from .state import State

TOL = 1e-3


def dispatch(sd: SystemData, st: State, min_line, cap_line):
    """由充放电反算火电出力、弃电、缺电。

    火电 + 储能放电 = 净负荷 + 储能充电 + 弃电 − 缺电
    """
    raw = sd.net_load + st.ch_total - st.dis_total
    curt = np.maximum(min_line - raw, 0.0)
    short = np.maximum(raw - cap_line, 0.0)
    gen = np.clip(raw, min_line, np.maximum(cap_line, min_line))
    return gen, curt, short


def validate(sd: SystemData, st: State, min_line, cap_line, guarantee_mask, ring: bool) -> dict:
    gen, curt, short = dispatch(sd, st, min_line, cap_line)
    checks = {}

    def add(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": detail}

    bal = gen + st.dis_total - (sd.net_load + st.ch_total + curt - short)
    add("逐时电量平衡", np.abs(bal).max() < TOL, f"最大残差 {np.abs(bal).max():.2e} MW")

    out = short[~guarantee_mask]
    add("非保供周火电不越容量线", (out < TOL).all(), f"最大越限 {out.max(initial=0):.3f} MW")
    gs = short[guarantee_mask].sum()
    add("保供周缺电与记录一致", abs(gs - st.shortage.sum()) < max(TOL, 1e-6 * gs),
        f"反算 {gs:.1f} MWh / 记录 {st.shortage.sum():.1f} MWh")
    thermal_ch = (sd.net_load + st.ch_total > min_line + TOL) & (st.ch_total > TOL)
    add("火电充电仅出现在保供周", not (thermal_ch & ~guarantee_mask).any(),
        f"{int((thermal_ch & ~guarantee_mask).sum())} 个小时")

    for j, s in enumerate(sd.storages):
        lift0 = st.lift[j, -1] if ring else 0.0
        rec = np.empty(sd.T)
        prev = lift0
        for t in range(sd.T):
            prev = s.rho * prev + s.eta_c * st.pch[j, t] - st.pdis[j, t] / s.eta_d
            rec[t] = prev
        err = np.abs(rec - st.lift[j]).max()
        add(f"[{s.name}] SOC 递推一致", err < TOL * 10, f"最大误差 {err:.2e} MWh")
        add(f"[{s.name}] SOC 在 [下限, 上限] 内",
            rec.min() > -TOL * 10 and rec.max() < s.usable + TOL * 10,
            f"范围 [{rec.min():.3f}, {rec.max():.3f}] / 可用 {s.usable:.1f} MWh")
        add(f"[{s.name}] 年末 SOC = 年初 SOC", abs(rec[-1] - lift0) < TOL * 10,
            f"年初 {lift0:.3f} / 年末 {rec[-1]:.3f} MWh")
        add(f"[{s.name}] 功率不越限",
            st.pch[j].max() <= s.power + TOL and st.pdis[j].max() <= s.power + TOL)
        add(f"[{s.name}] 不同时充放", not ((st.pch[j] > TOL) & (st.pdis[j] > TOL)).any())
    return checks
