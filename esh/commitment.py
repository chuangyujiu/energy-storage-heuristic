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
        self.maint = self._schedule_maintenance() if self.cfg.maintenance else None

    def order(self, week: int, prev_on: set | None = None) -> list[int]:
        """开机优先级：本周必开机组在前，其余按煤耗从低到高。

        startup_aware 时按单位电量成本排序：燃料成本（标煤单价 × 标杆煤耗 × 煤耗系数）
        + 前一周期未开机组的启动成本分摊（启停成本 / (单机容量 × 分摊小时数)），元/MWh。
        """
        must = [i for i, u in enumerate(self.units) if u.is_must_run(week)]
        rest = [i for i, u in enumerate(self.units) if not u.is_must_run(week)]
        if self.cfg.startup_aware:
            cfg = self.cfg
            fuel = cfg.coal_price * cfg.base_heat_rate / 1e3

            def key(i):
                u = self.units[i]
                c = fuel * u.heat_coef
                if prev_on is not None and i not in prev_on:
                    c += u.startup * cfg.startup_yuan / (u.capacity * cfg.startup_spread_hours)
                return (c, -u.capacity, i)
        else:
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

    def _schedule_maintenance(self) -> np.ndarray:
        """等备用法安排检修，返回 (机组, 周期) 的检修标记。

        每个开机周期的火电备用 = 可用容量 − max(净负荷 + 备用需求)。从大机组开始逐台安排：
        检修时长按周期换算（向上取整），在同时检修容量不超过 maint_max_ratio × 总装机的前提下，
        选检修期间最小备用最大的连续时段（并列取最早），排完扣除该机组容量。
        """
        from numpy.lib.stride_tricks import sliding_window_view as win
        bounds = self.sd.week_bounds()
        P, n = len(bounds), len(self.units)
        cfg, nl, load = self.cfg, self.sd.net_load, self.sd.load
        need, mon = np.zeros(P), np.zeros(P, int)
        for p, (a, b) in enumerate(bounds):
            res = load[a:b] * cfg.load_reserve_rate + load[a:b] * cfg.accident_reserve_rate
            need[p] = (nl[a:b] + np.maximum(res, cfg.min_reserve)).max()
            mon[p] = self.month[(a + b) // 2]
        avail_u = self.cap[:, None] * self.ratio[:, mon]              # (机组, 周期)
        margin = avail_u.sum(0) - need
        out = np.zeros(P)
        limit = cfg.maint_max_ratio * self.cap.sum()
        mask = np.zeros((n, P), bool)
        hours = self.sd.config.commit_hours
        for u in np.argsort(-self.cap, kind="stable"):
            days = self.units[u].maint_days
            if days <= 0:
                continue
            L = min(P, int(np.ceil(days * 24 / hours)))
            ok = win(out, L).max(1) + self.cap[u] <= limit + 1e-6
            if not ok.any():
                continue
            score = win(margin - avail_u[u], L).min(1)
            score[~ok] = -np.inf
            s = int(np.argmax(score))
            mask[u, s:s + L] = True
            margin[s:s + L] -= avail_u[u, s:s + L]
            out[s:s + L] += self.cap[u]
        return mask

    def initial(self, week: int, a: int, b: int, prev_on: set | None = None):
        """初始开机台数 N 与初始目标 k₀ = 按天最大开机台数 − 最小开机台数。

        week 为锁定周期序号；必开机组（如供暖季）按该周期起点所在的周判断；检修中的机组不参与开机。
        """
        period = week
        week = min(a // 168, 51)
        order = self.order(week, prev_on)
        if self.maint is not None:
            order = [u for u in order if not self.maint[u, period]]
        nm = self.n_floor(week, order)
        N = self.required(order, np.arange(a, b), nm)
        daily = [self.required(order, np.arange(s, min(s + 24, b)), nm) for s in range(a, b, 24)]
        k0 = max(daily) - min(daily)
        return order, N, k0, N - min(nm, N)
