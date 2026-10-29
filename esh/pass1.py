"""第一遍：容量效益——逐周找出"全周可行地少开 k 台"的最大 k。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commitment import Fleet
from .model import EPS, SystemData, storage_order
from .state import State
from .waterfill import waterfill

TOL = 1e-6


@dataclass
class WeekResult:
    week: int
    start: int
    end: int
    N: int          # 初始开机台数
    k0: int         # 初始目标
    k: int          # 最终少开台数
    k_mw: float     # 少开机组容量 MW
    guarantee: bool  # 是否触发保供模式
    shortage: float  # 缺电量 MWh
    trials: int      # 试算次数


class Pass1:
    """ring_prev：环形年时上一轮的结果（提供 12 月末的开机线与已用弃电）。"""

    def __init__(self, sd: SystemData, ring_prev: "Pass1 | None" = None):
        self.sd = sd
        self.T = sd.T
        self.nl = sd.net_load
        self.fleet = Fleet(sd)
        self.st = State(self.T, len(sd.storages))
        self.order_s = storage_order(sd.storages)
        self.cap_line = np.full(self.T, np.nan)
        self.min_line = np.full(self.T, np.nan)
        self.pre_used = np.zeros(self.T)
        self.ring_prev = ring_prev
        self.ring = ring_prev is not None
        self.weeks: list[WeekResult] = []
        self.commit: list[list[int]] = []    # 每周在运机组

    # ---------------------------------------------------------------- 配对
    def _avail(self, idx: np.ndarray, line: np.ndarray) -> np.ndarray:
        """可充电空间 = 充电上限线 − 净负荷 − 已被占用部分；有放电的小时不充电。"""
        a = line[idx] - self.nl[idx] - self.st.ch_total[idx] - self.pre_used[idx]
        a = np.where(np.isnan(a) | (self.st.dis_total[idx] > EPS), 0.0, a)
        return np.maximum(a, 0.0)

    def _pair(self, d: int, R: float, line: np.ndarray, tag: str) -> float:
        for j in self.order_s:
            if R <= EPS:
                break
            R = self._pair_one(j, d, R, line, tag)
        return R

    def _pair_one(self, j: int, d: int, R0: float, line: np.ndarray, tag: str) -> float:
        """为第 d 小时的削峰需求 R0 在储能 j 上往回找充电位置，返回未满足部分。"""
        s, st, T = self.sd.storages[j], self.st, self.T
        R = min(R0, s.power - st.pdis[j, d])
        if R <= EPS:
            return R0
        rest = R0 - R
        block = -10 ** 9                      # SOC 已顶满的最晚时刻，充电位置必须在其之后
        for w in s.windows:                   # 前瞻窗口逐步扩大
            lo = d - w if self.ring else max(0, d - w)
            offs = np.arange(lo, d)
            while R > EPS:
                o = offs[offs > block]
                if o.size == 0:
                    break
                idx = o % T
                av = self._avail(idx, line)
                pc = s.power - st.pch[j, idx]
                ok = (av > EPS) & (pc > EPS) & (st.pdis[j, idx] <= EPS)
                if not ok.any():
                    break
                o, idx, av, pc = o[ok], idx[ok], av[ok], pc[ok]
                # 放电端已知，按 E_dis = E_ch·η_c·(1−σ)^h·η_d 反算充电端
                g = s.eta_c * s.eta_d * s.rho ** (d - o)
                x = waterfill(av, pc, g, R, floor=0.0)
                if x.sum() <= EPS:
                    break
                # 区间 [C 开始, D 结束) 内的 SOC 抬高量
                base = int(o[0])
                tau = np.arange(base, d)
                inflow = np.zeros(tau.size)
                inflow[o - base] = x * s.eta_c * s.rho ** (-(o - base))
                lift = np.cumsum(inflow) * s.rho ** (tau - base)
                tidx = tau % T
                H = np.maximum(s.usable - st.lift[j, tidx], 0.0)
                pos = lift > EPS
                ratio = np.full(tau.size, np.inf)
                ratio[pos] = H[pos] / lift[pos]
                alpha = min(1.0, float(ratio.min()))
                if alpha > 1e-12:
                    xa = alpha * x
                    e_dis = float(np.dot(xa, g))
                    st.pch[j, idx] += xa
                    st.ch_total[idx] += xa
                    st.lift[j, tidx] += alpha * lift
                    st.pdis[j, d] += e_dis
                    st.dis_total[d] += e_dis
                    R -= e_dis
                    for c, e in zip(o[xa > EPS], xa[xa > EPS]):
                        st.pairs.append((j, int(c % T), d, float(e),
                                         float(e * s.eta_c * s.eta_d * s.rho ** (d - c)), tag))
                if alpha < 1.0 - 1e-12:
                    # 容量受限：把电量缩到刚好满足，剩余部分只能在顶满时刻之后充电
                    sat = np.nonzero(pos & (ratio <= alpha * (1 + 1e-9) + 1e-12))[0]
                    block = int(tau[sat[-1]])
                else:
                    break                     # 本窗口的弃电或功率已用尽 → 扩大窗口
            if R <= EPS:
                break
        return max(R, 0.0) + rest

    # ---------------------------------------------------------------- 试算
    def _trial(self, wk: int, a: int, b: int, order: list[int], n: int, guarantee: bool) -> bool:
        hours = np.arange(a, b)
        cap, mn = self.fleet.lines(order, n, hours)
        self.cap_line[a:b], self.min_line[a:b] = cap, mn
        st = self.st
        # 环形年：12 月已被跨年配对预留的充电必须仍落在弃电内
        reserved = st.ch_total[a:b] > EPS
        if reserved.any() and (self.nl[a:b] + st.ch_total[a:b] - mn > TOL)[reserved].any():
            return False
        demand = np.maximum(self.nl[a:b] - cap, 0.0)   # 净负荷高于容量线 → 削峰需求（放电全部确定）
        unmet = {}
        for i in np.nonzero(demand > EPS)[0]:          # 按时间从早到晚找充电
            d = a + int(i)
            R = self._pair(d, float(demand[i]), self.min_line, "p1")
            if R > EPS:
                unmet[d] = R
        if not unmet:
            return True
        if not guarantee:
            return False
        # 保供模式：只对未配满的需求放开火电充电，上限提高到全开机组容量线
        for d, R in sorted(unmet.items()):
            R = self._pair(d, R, self.cap_line, "p1-thermal")
            if R > EPS:
                st.shortage[d] += R
        return True

    def _setup_ring(self, on: bool):
        """第 1 周往回找时可绕回 12 月末的剩余弃电（日储能 3 天、周储能 7 天）。"""
        if not self.ring:
            return
        w = slice(self.T - 168, self.T)
        if on:
            p = self.ring_prev
            self.cap_line[w], self.min_line[w] = p.cap_line[w], p.min_line[w]
            self.pre_used[w] = p.st.ch_total[w]
        else:
            self.cap_line[w] = self.min_line[w] = np.nan
            self.pre_used[w] = 0.0

    def run(self) -> "Pass1":
        st = self.st
        for wk, (a, b) in enumerate(self.sd.week_bounds()):
            if wk == 0:
                self._setup_ring(True)
            order, N, k0, kmax = self.fleet.initial(wk, a, b)
            k0 = int(np.clip(k0, 0, kmax))
            s0 = st.snapshot()
            trials = 0

            def attempt(k, guarantee=False):
                nonlocal trials
                trials += 1
                st.restore(s0)
                return self._trial(wk, a, b, order, N - k, guarantee)

            best, snap = None, None
            if attempt(k0):
                best, snap = k0, st.snapshot()
                k = k0 + 1
                while k <= kmax and attempt(k):
                    best, snap = k, st.snapshot()
                    k += 1
            else:
                for k in range(k0 - 1, -1, -1):
                    if attempt(k):
                        best, snap = k, st.snapshot()
                        break
            guarantee = best is None
            if guarantee:
                attempt(0, guarantee=True)
                best, snap = 0, st.snapshot()
            st.restore(snap)
            if wk == 0:
                self._setup_ring(False)
            hours = np.arange(a, b)
            self.cap_line[a:b], self.min_line[a:b] = self.fleet.lines(order, N - best, hours)
            stopped = order[N - best:N]
            self.commit.append(order[:N - best])
            self.weeks.append(WeekResult(wk, a, b, N, k0, best,
                                         float(self.fleet.cap[stopped].sum()), guarantee,
                                         float(st.shortage[a:b].sum()), trials))
        return self

    @property
    def guarantee_mask(self) -> np.ndarray:
        m = np.zeros(self.T, bool)
        for w in self.weeks:
            m[w.start:w.end] = w.guarantee
        return m
