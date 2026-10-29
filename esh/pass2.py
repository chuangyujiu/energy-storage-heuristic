"""第二遍：消纳——用剩余弃电和储能空间吸收弃电，放电截止于下一段弃电。

开机方式已由第一遍确定。按时间顺序取出每一段剩余弃电，储能按"短时优先、同类内
效率优先"依次在 SOC 与功率允许范围内尽量充电；放电区间为本段弃电结束到下一段弃电
开始，区间内对火电出力做削峰灌水，每小时不超过放电空间（火电出力 − 最小出力线）；
放不完的电量延到下一个区间。

自损耗是线性的，故用"贴现电量"记账：t 时刻的 1 MWh 记为 ρ^(−t) MWh，
充入与放出的贴现电量相等即闭合。
"""
from __future__ import annotations

import numpy as np

from .model import EPS, SystemData, storage_order
from .state import State
from .waterfill import waterfill

TOL = 1e-6


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """连续为 True 的区段 [start, end]（含端点）。"""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


class Pass2:
    def __init__(self, sd: SystemData, st: State, min_line: np.ndarray, cap_line: np.ndarray):
        self.sd, self.st = sd, st
        self.min_line, self.cap_line = min_line, cap_line
        self.nl = sd.net_load
        self.T = sd.T
        self.order_s = storage_order(sd.storages)
        self.transfers: list[dict] = []

    def _gen(self, idx):
        """当前（未截断的）火电出力 = 净负荷 + 充电 − 放电。"""
        return self.nl[idx] + self.st.ch_total[idx] - self.st.dis_total[idx]

    def run(self) -> "Pass2":
        cur = np.maximum(self.min_line - self._gen(slice(None)), 0.0)
        segs = _segments(cur > EPS)
        # 放电区间：第 i 段弃电结束 → 第 i+1 段弃电开始（末段到年末）
        gaps = [(segs[i][1] + 1, (segs[i + 1][0] - 1) if i + 1 < len(segs) else self.T - 1)
                for i in range(len(segs))]
        for i, (s0, s1) in enumerate(segs):
            for j in self.order_s:
                self._transfer(j, s0, s1, gaps[i:], i)
        return self

    def _plan(self, j, S, caps_c, gaps, A):
        s, st = self.sd.storages[j], self.st
        s0 = int(S[0])
        cur = np.maximum(self.min_line[S] - self._gen(S), 0.0)
        x = waterfill(cur, caps_c, np.ones(S.size), A, floor=0.0)   # 弃电深度灌水
        pv_in = x * s.eta_c * s.rho ** (-(S - s0))
        rem = float(pv_in.sum())
        dis = []
        for i0, i1 in gaps:
            if rem <= EPS * 1e-3:
                break
            if i1 < i0:
                continue
            D = np.arange(i0, i1 + 1)
            g = self._gen(D)
            cap = np.minimum(g - self.min_line[D], s.power - st.pdis[j, D])
            cap = np.where(st.ch_total[D] > EPS, 0.0, np.maximum(cap, 0.0))
            wgt = s.rho ** (-(D - s0)) / s.eta_d
            tot = float(np.dot(cap, wgt))
            if tot <= rem:
                y = cap
                rem -= tot
            else:
                y = waterfill(g, cap, wgt, rem)               # 削峰灌水
                rem = 0.0
            dis.append((D, y, wgt))
        if rem > TOL or not dis:
            return None
        last = int(dis[-1][0][-1])
        tau = np.arange(s0, last + 1)
        flow = np.zeros(tau.size)
        flow[S - s0] += pv_in
        for D, y, wgt in dis:
            flow[D - s0] -= y * wgt
        lift = np.maximum(np.cumsum(flow) * s.rho ** (tau - s0), 0.0)
        lift[-1] = 0.0 if abs(lift[-1]) < 1e-6 else lift[-1]
        if (st.lift[j, tau] + lift > s.usable + TOL).any():
            return None
        return x, dis, tau, lift

    def _transfer(self, j, s0, s1, gaps, seg_id):
        s, st = self.sd.storages[j], self.st
        S = np.arange(s0, s1 + 1)
        cur = np.maximum(self.min_line[S] - self._gen(S), 0.0)
        caps = np.where(st.pdis[j, S] > EPS, 0.0, np.maximum(s.power - st.pch[j, S], 0.0))
        caps = np.minimum(caps, cur)
        H = s.usable - st.lift[j, S]
        a_hi = min(float(caps.sum()),
                   max(float(H.max()), 0.0) / s.eta_c / s.rho ** S.size)
        if a_hi <= EPS:
            return
        plan = self._plan(j, S, caps, gaps, a_hi)
        if plan is None:                      # 在 SOC 与功率允许的范围内尽量充电（二分电量）
            lo, hi, plan = 0.0, a_hi, None
            while hi - lo > max(1e-4 * a_hi, 1e-3):
                mid = 0.5 * (lo + hi)
                p = self._plan(j, S, caps, gaps, mid)
                if p is None:
                    hi = mid
                else:
                    lo, plan = mid, p
            if plan is None:
                return
        x, dis, tau, lift = plan
        st.pch[j, S] += x
        st.ch_total[S] += x
        e_dis = 0.0
        for D, y, _ in dis:
            st.pdis[j, D] += y
            st.dis_total[D] += y
            e_dis += float(y.sum())
        st.lift[j, tau] += lift
        if x.sum() > EPS:
            self.transfers.append(dict(segment=seg_id, storage=s.name, start=int(s0), end=int(s1),
                                       charge=float(x.sum()), discharge=e_dis,
                                       last_discharge_hour=int(tau[-1])))
