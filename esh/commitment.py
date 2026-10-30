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
        # 分月可用容量比例（供暖季降出力等）：shape (机组, 12)
        self.ratio = np.array([u.max_ratio if u.max_ratio else (1.0,) * 12 for u in self.units])
        self.month = sd.month
        self._cache: dict = {}

    def order(self, week: int) -> list[int]:
        """开机优先级：本周必开机组在前，其余按煤耗从低到高。"""
        must = [i for i, u in enumerate(self.units) if u.is_must_run(week)]
        rest = [i for i, u in enumerate(self.units) if not u.is_must_run(week)]
        key = lambda i: self.units[i].coal
        return sorted(must, key=key) + sorted(rest, key=key)

    def n_must(self, week: int) -> int:
        return sum(u.is_must_run(week) for u in self.units)

    def n_floor(self, week: int, order: list[int]) -> int:
        """开机台数下限：必开机组，以及保安开机（按优先级开到在运容量 ≥ 比例 × 总装机）。"""
        n = self.n_must(week)
        need = self.cfg.min_online_ratio * self.cap.sum()
        if need > 0:
            cum = np.cumsum(self.cap[order])
            n = max(n, int(np.searchsorted(cum, need - 1e-9)) + 1)
        return min(n, len(order))

    def _cum(self, order: list[int]) -> dict:
        """按开机顺序累加：前 n 台的分月可用容量、分月最小出力、最大单机（缓存）。"""
        key = tuple(order)
        c = self._cache.get(key)
        if c is None:
            o = np.asarray(order, int)
            avail = self.cap[o, None] * self.ratio[o]
            pmin = np.minimum(self.pmin[o, None], avail)
            z = np.zeros((1, 12))
            c = {"avail": np.vstack([z, np.cumsum(avail, 0)]),
                 "pmin": np.vstack([z, np.cumsum(pmin, 0)]),
                 "big": np.concatenate([[0.0], np.maximum.accumulate(self.cap[o])]) if o.size
                 else np.zeros(1)}
            self._cache[key] = c
        return c

    def lines(self, order: list[int], n: int, hours: np.ndarray):
        """按优先级开前 n 台：返回 (容量线, 最小出力线)。

        负荷备用 = 负荷 × 负荷备用率；
        事故备用 = max(负荷 × 事故备用率, 在运最大单机容量)（可关闭最大单机项）；
        总备用不低于最小备用。容量按分月最大出力比例折算。
        """
        cum = self._cum(order)
        load = self.sd.load[hours]
        biggest = cum["big"][n] if self.cfg.accident_largest_unit else 0.0
        reserve = load * self.cfg.load_reserve_rate + np.maximum(
            load * self.cfg.accident_reserve_rate, biggest)
        reserve = np.maximum(reserve, self.cfg.min_reserve)
        mon = self.month[hours]
        cap_line = cum["avail"][n][mon] - reserve       # 各月在运可用容量 − 备用
        min_line = cum["pmin"][n][mon]
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
        order = self.order(week)
        nm = self.n_floor(week, order)
        N = self.required(order, np.arange(a, b), nm)
        daily = [self.required(order, np.arange(s, min(s + 24, b)), nm) for s in range(a, b, 24)]
        k0 = max(daily) - min(daily)
        return order, N, k0, N - min(nm, N)
