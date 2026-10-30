"""第三遍（可选）：弃电不足时，允许火电在低谷时段给储能充电，在高峰时段放电。

在第二遍之后进行，开机方式沿用第一遍结果。每个调度窗口（默认每天 10 时至次日 10 时）
内，对每种储能：
- 充电只落在没有剩余弃电的时段，充电后火电出力不超过容量线（保留备用）；
- 放电后火电出力不低于最小出力线（不制造新的弃电）；
- 充电按火电出力"填谷"灌水，放电按火电出力"削峰"灌水；
- 火电边际成本随出力升高：低谷充电后的水位 ≤ 比例 × 高峰放电后的水位才继续，
  比例默认取该储能的往返效率 η（峰谷差需大于储能损耗）。
电量按 ρ^(−t) 贴现记账，与第二遍相同。
"""
from __future__ import annotations

import numpy as np

from .model import EPS, SystemData, storage_order
from .state import State
from .waterfill import waterfill

TOL = 1e-6


class Pass3:
    def __init__(self, sd: SystemData, st: State, min_line: np.ndarray, cap_line: np.ndarray):
        self.sd, self.st = sd, st
        self.min_line, self.cap_line = min_line, cap_line
        self.nl = sd.net_load
        self.T = sd.T
        self.order_s = storage_order(sd.storages)
        self.thermal_charge = np.zeros(self.T)   # 本遍火电充电功率（校验用）
        self.transfers: list[dict] = []

    def _gen(self, idx):
        return self.nl[idx] + self.st.ch_total[idx] - self.st.dis_total[idx]

    def run(self) -> "Pass3":
        start = int(self.sd.config.valley_window_start) % 24
        for a in range(start, self.T, 24):
            W = np.arange(a, min(a + 24, self.T))
            if W.size < 2:
                continue
            for j in self.order_s:
                self._shift(j, W)
        return self

    def _plan(self, j, W, A):
        s, st = self.sd.storages[j], self.st
        g = self._gen(W)
        ratio = self.sd.config.valley_ratio or s.eta_rt
        cap_c = np.where(st.pdis[j, W] > EPS, 0.0,
                         np.minimum(s.power - st.pch[j, W], self.cap_line[W] - g))
        cap_c = np.where(self.min_line[W] - g > EPS, 0.0, np.maximum(cap_c, 0.0))   # 有弃电的时段留给第二遍
        x = waterfill(-g, cap_c, np.ones(W.size), A)              # 填谷
        pv = s.rho ** (-(W - W[0]))
        pv_in = float(np.dot(x, pv)) * s.eta_c
        cap_d = np.minimum(g - self.min_line[W], s.power - st.pdis[j, W])
        cap_d = np.where((st.pch[j, W] > EPS) | (x > EPS), 0.0, np.maximum(cap_d, 0.0))
        if (x > EPS).any():                                        # 先充后放：放电只在最后一个充电小时之后
            cap_d[: int(np.nonzero(x > EPS)[0][-1]) + 1] = 0.0
        wgt = pv / s.eta_d
        if float(np.dot(cap_d, wgt)) < pv_in - TOL:
            return None
        y = waterfill(g, cap_d, wgt, pv_in)                        # 削峰
        flow = x * s.eta_c * pv - y * wgt
        lift = np.cumsum(flow) * s.rho ** (W - W[0])
        if lift.min() < -1e-6 or abs(lift[-1]) > 1e-6:
            return None                                            # 先放后充或窗口内未放完
        if (st.lift[j, W] + lift > s.usable + TOL).any():
            return None
        ch, dh = x > EPS, y > EPS
        if ch.any() and dh.any():
            valley = float((g + x)[ch].max())
            peak = float((g - y)[dh].min())
            if valley > ratio * peak + TOL:
                return None                                        # 峰谷差不足以弥补损耗
        return x, y, np.maximum(lift, 0.0)

    def _shift(self, j, W):
        s, st = self.sd.storages[j], self.st
        hi = min(s.power * W.size, s.usable / s.eta_c)
        if hi <= EPS or self._plan(j, W, 1e-3) is None:
            return
        plan = self._plan(j, W, hi)
        if plan is None:
            lo, plan = 0.0, None
            while hi - lo > max(1e-4 * hi, 1e-2):
                mid = 0.5 * (lo + hi)
                p = self._plan(j, W, mid)
                if p is None:
                    hi = mid
                else:
                    lo, plan = mid, p
            if plan is None:
                return
        x, y, lift = plan
        if x.sum() <= EPS:
            return
        st.pch[j, W] += x
        st.ch_total[W] += x
        st.pdis[j, W] += y
        st.dis_total[W] += y
        st.lift[j, W] += lift
        self.thermal_charge[W] += x
        self.transfers.append(dict(storage=s.name, start=int(W[0]), charge=float(x.sum()),
                                   discharge=float(y.sum())))
