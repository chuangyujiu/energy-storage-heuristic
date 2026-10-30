"""火电开机：优先级、备用、容量线与最小出力线。"""
from __future__ import annotations

import numpy as np

from .model import SystemData


class Fleet:
    def __init__(self, sd: SystemData):
        self.sd = sd
        self.units = sd.units
        self.cfg = sd.config
        self.cap = np.array([u.capacity for u in self.units])
        self.pmin = np.array([u.pmin for u in self.units])

    def order(self, week: int) -> list[int]:
        """开机优先级：本周必开机组在前，其余按煤耗从低到高。"""
        must = [i for i, u in enumerate(self.units) if u.is_must_run(week)]
        rest = [i for i, u in enumerate(self.units) if not u.is_must_run(week)]
        key = lambda i: self.units[i].coal
        return sorted(must, key=key) + sorted(rest, key=key)

    def n_must(self, week: int) -> int:
        return sum(u.is_must_run(week) for u in self.units)

    def lines(self, order: list[int], n: int, hours: np.ndarray):
        """按优先级开前 n 台：返回 (容量线, 最小出力线)。

        负荷备用 = 负荷 × 负荷备用率；
        事故备用 = max(负荷 × 事故备用率, 在运最大单机容量)。
        """
        on = order[:n]
        load = self.sd.load[hours]
        biggest = self.cap[on].max() if on else 0.0
        reserve = load * self.cfg.load_reserve_rate + np.maximum(
            load * self.cfg.accident_reserve_rate, biggest)
        cap_line = self.cap[on].sum() - reserve
        min_line = np.full(len(hours), self.pmin[on].sum())
        return cap_line, min_line

    def required(self, order: list[int], hours: np.ndarray, n_must: int) -> int:
        """不考虑储能，按优先级依次开机，直到每小时都满足"净负荷 + 备用"。

        火电出力不低于最小出力线，备用也要在其之上留出，故要求
        容量线 ≥ max(净负荷, 最小出力线)。
        """
        nl = self.sd.net_load[hours]
        for n in range(max(n_must, 1), len(order) + 1):
            cap_line, min_line = self.lines(order, n, hours)
            if (cap_line >= np.maximum(nl, min_line)).all():
                return n
        return len(order)

    def initial(self, week: int, a: int, b: int):
        """初始开机台数 N 与初始目标 k₀ = 按天最大开机台数 − 最小开机台数。"""
        order, nm = self.order(week), self.n_must(week)
        N = self.required(order, np.arange(a, b), nm)
        daily = [self.required(order, np.arange(s, min(s + 24, b)), nm) for s in range(a, b, 24)]
        k0 = max(daily) - min(daily)
        return order, N, k0, N - min(nm, N)
