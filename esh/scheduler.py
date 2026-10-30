"""算法总流程：第一遍（容量效益）→ 第二遍（消纳）→ [第三遍（火电低谷充电）] → 校验与指标。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commitment import Fleet
from .model import SystemData
from .pass1 import Pass1
from .pass2 import Pass2
from .pass3 import Pass3
from .validate import dispatch, validate


@dataclass
class Result:
    mode: str
    p1: Pass1
    p2: Pass2
    stages: dict           # 各阶段的 (火电出力, 弃电, 缺电)
    checks: dict
    ring_iters: int = 0
    p3: Pass3 | None = None

    @property
    def st(self):
        return self.p1.st


def _baseline(sd: SystemData):
    """储能投入前：每周按初始开机 N 运行。"""
    fleet = Fleet(sd)
    cap = np.empty(sd.T)
    mn = np.empty(sd.T)
    for wk, (a, b) in enumerate(sd.week_bounds()):
        order, N, _, _ = fleet.initial(wk, a, b)
        cap[a:b], mn[a:b] = fleet.lines(order, N, np.arange(a, b))
    raw = sd.net_load
    return (np.clip(raw, mn, np.maximum(cap, mn)), np.maximum(mn - raw, 0.0),
            np.maximum(raw - cap, 0.0))


def _ring_converged(a: Pass1, b: Pass1) -> bool:
    w = slice(a.T - 168, a.T)
    return ([x.k for x in a.weeks[-2:]] == [x.k for x in b.weeks[-2:]]
            and np.allclose(a.st.ch_total[w], b.st.ch_total[w], atol=1e-3))


def solve(sd: SystemData, mode: str = "base") -> Result:
    """mode = "base"（年初 SOC 取下限，不跨年）或 "ring"（环形年对比方案）。"""
    p1 = Pass1(sd).run()
    iters = 0
    if mode == "ring":
        for iters in range(1, sd.config.ring_max_iter + 1):
            nxt = Pass1(sd, ring_prev=p1).run()
            done = _ring_converged(p1, nxt)
            p1 = nxt
            if done:
                break
    ring = mode == "ring"
    stages = {"before": _baseline(sd)}
    stages["pass1"] = dispatch(sd, p1.st, p1.min_line, p1.cap_line)
    p2 = Pass2(sd, p1.st, p1.min_line, p1.cap_line).run()
    stages["pass2"] = dispatch(sd, p1.st, p1.min_line, p1.cap_line)
    # 允许火电充电的小时：保供模式实际用火电充电的小时（窗口可跨入前一周期），以及第三遍充电时段
    thermal_ok = np.zeros(sd.T, bool)
    thermal_ok[[c for _, c, _, _, _, tag in p1.st.pairs if tag == "p1-thermal"]] = True
    p3 = None
    if sd.config.thermal_valley_charging:
        p3 = Pass3(sd, p1.st, p1.min_line, p1.cap_line).run()
        stages["pass3"] = dispatch(sd, p1.st, p1.min_line, p1.cap_line)
        thermal_ok |= p3.thermal_charge > 1e-6
    checks = validate(sd, p1.st, p1.min_line, p1.cap_line, p1.guarantee_mask, ring, thermal_ok)
    return Result(mode, p1, p2, stages, checks, iters, p3)
