"""用 GTDS 的火电开机方案作为输入，本算法只做储能调度。

1. 由 8760h 电力平衡表中各火电厂的逐时出力，反推每厂每天的开机台数
   （GTDS 按日启停：开停只发生在 0 时，当天台数不变）：
   - 至少台数 = ceil(当天最大出力 / 单机可用容量)
   - 最多台数 = floor(当天最小出力 / 单机最小出力)
   - 当天最小出力恰为 k 台最小出力之和（机组压到最低）时取 k；否则两者相等即确定；
     仍不确定时取至少台数
2. 最小出力线 = Σ 台数 × 单机最小出力，可用容量 = Σ 台数 × 单机可用容量（逐时，按月折算）
3. 开机固定：第一遍不搜索 k，只对净负荷高于容量线的时段配对放电（配不满时火电充电、记缺电），
   之后照常第二遍（消纳）、可选第三遍。
"""
from __future__ import annotations

import re

import numpy as np
import openpyxl
import pandas as pd

from .balance import _table, read_balance
from .pass1 import EPS, Pass1, WeekResult

THERMAL = 300


def infer_commitment(balance_xlsx, params_xlsx, system: str = "西北"):
    """返回 (每厂每日台数表, 逐时可用容量, 逐时最小出力线, 统计)。"""
    _, rows = read_balance(balance_xlsx, system)
    i0 = next(k for k, (lab, _) in enumerate(rows) if lab == "六、火电出力合计")
    plants = [(re.sub(r"^\d+\)\s*", "", lab)[:-2].strip(), v) for lab, v in rows[i0 + 2:]
              if lab.endswith("发电")]
    wb = openpyxl.load_workbook(params_xlsx, data_only=True)
    units = _table(wb, "机组表").drop(columns=["类型"])
    ptab = _table(wb, "电站表")
    units = units[units["有效性"] == 1].merge(ptab[["电站ID", "类型"]], on="电站ID")
    units = units[units["类型"] == THERMAL].copy()
    units["名"] = units["名称"].str.replace(r"^U\d+", "", regex=True).str.strip()
    cons = _table(wb, "机组约束")
    cmax = cons[(cons["类型"] == 0) & (cons["有效性"] == 1)].set_index("机组ID")

    T = len(plants[0][1])
    days = T // 24
    month = np.minimum(np.searchsorted(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24,
                                       np.arange(T), side="right") - 1, 11)
    avail, pmin_line = np.zeros(T), np.zeros(T)
    used, recs = set(), []
    stat = {"厂日_开机": 0, "厂日_唯一确定": 0, "厂日_按最低出力确定": 0, "厂日_取至少台数": 0,
            "厂日_超出可行范围": 0}
    for name, v in plants:
        u = units[(units["名"] == name) & ~units["机组ID"].isin(used)]
        if len(u) == 0:
            continue
        u = u.iloc[0]
        used.add(u["机组ID"])
        n, cap, tech = int(u["台数"]), float(u["单机容量"]), min(float(u["技术出力"]), 1.0)
        if n <= 0:
            continue
        ratio = (np.array([float(cmax.loc[u["机组ID"], f"M{m + 1}"]) for m in range(12)])
                 if u["机组ID"] in cmax.index else np.ones(12))
        pmax = cap * ratio[month]
        pmn = np.minimum(cap * tech, pmax)
        vd, pxd, pnd = v.reshape(days, 24), pmax.reshape(days, 24), pmn.reshape(days, 24)
        k = np.zeros(days, int)
        for d in range(days):
            if vd[d].max() <= 1:
                continue
            stat["厂日_开机"] += 1
            lo = int(np.ceil(np.round((vd[d] / pxd[d]).max(), 6)))
            hi = int(np.floor(np.round((vd[d] / pnd[d]).min(), 6) + 1e-9)) if tech > 0 else n
            hi = min(hi, n)
            kk = vd[d].min() / pnd[d][vd[d].argmin()] if tech > 0 else 0
            if tech > 0 and abs(kk - round(kk)) < 1e-3 and lo <= round(kk) <= hi:
                k[d] = int(round(kk))
                stat["厂日_按最低出力确定"] += 1
            elif lo == hi:
                k[d] = lo
                stat["厂日_唯一确定"] += 1
            else:
                k[d] = min(max(lo, 1), n)
                stat["厂日_超出可行范围" if lo > hi else "厂日_取至少台数"] += 1
        kh = np.repeat(k, 24)
        avail += kh * pmax
        pmin_line += kh * pmn
        recs.append(dict(电厂=name, 台数=n, 单机容量=cap, 技术出力=tech, **{f"D{d + 1}": k[d] for d in range(days)}))
    stat["电厂数"] = len(recs)
    return pd.DataFrame(recs), avail, pmin_line, stat


class FixedPass1(Pass1):
    """开机方式固定（逐时可用容量与最小出力线给定）的第一遍：只为削峰需求配对。"""

    def __init__(self, sd, avail, pmin_line, ring_prev: "FixedPass1 | None" = None):
        super().__init__(sd, ring_prev)
        self.avail, self.pmin_line = avail, pmin_line

    def _wrap(self, d: int):
        """环形年：第 1 周的需求可绕回上一轮 12 月末的剩余弃电（扣除上一轮已用的充电）。"""
        if self.ring:
            w = slice(self.T - 168, self.T)
            self.pre_used[w] = self.ring_prev.st.ch_total[w] if d < 168 else 0.0

    def run(self) -> "FixedPass1":
        cfg, load = self.sd.config, self.sd.load
        reserve = load * cfg.load_reserve_rate + load * cfg.accident_reserve_rate
        reserve = np.maximum(reserve, cfg.min_reserve)
        self.cap_line[:] = self.avail - reserve
        self.min_line[:] = self.pmin_line
        demand = np.maximum(self.nl - self.cap_line, 0.0)
        unmet = {}
        for d in np.nonzero(demand > EPS)[0]:
            self._wrap(int(d))
            R = self._pair(int(d), float(demand[d]), self.min_line, "p1")
            if R > EPS:
                unmet[int(d)] = R
        for d, R in sorted(unmet.items()):          # 配不满：放开火电充电，仍不足记缺电
            self._wrap(d)
            R = self._pair(d, R, self.cap_line, "p1-thermal")
            if R > EPS:
                self.st.shortage[d] += R
        for i in range(self.T // 24):
            a, b = 24 * i, 24 * (i + 1)
            g = any(a <= d < b for d in unmet)
            self.weeks.append(WeekResult(i, a, b, 0, 0, 0, 0.0, g, float(self.st.shortage[a:b].sum()), 0))
        self._wrap(self.T)
        return self


def solve_fixed(sd, avail, pmin_line, ring: bool = False):
    """与 scheduler.solve 相同的后续流程（第二遍、可选第三遍、校验），开机固定为给定方案。

    ring=True 为环形年：先按基准方案跑第一遍，再让第 1 周绕回上一轮 12 月末的剩余弃电重跑，
    直到 12 月末的充电不再变化；跨年配对的电量即为年初 SOC。
    """
    from .pass2 import Pass2
    from .pass3 import Pass3
    from .scheduler import Result
    from .validate import dispatch, validate
    p1 = FixedPass1(sd, avail, pmin_line).run()
    iters = 0
    if ring:
        w = slice(sd.T - 168, sd.T)
        for iters in range(1, sd.config.ring_max_iter + 1):
            nxt = FixedPass1(sd, avail, pmin_line, ring_prev=p1).run()
            done = np.allclose(nxt.st.ch_total[w], p1.st.ch_total[w], atol=1e-3)
            p1 = nxt
            if done:
                break
    raw = sd.net_load
    stages = {"before": (np.clip(raw, p1.min_line, np.maximum(p1.cap_line, p1.min_line)),
                         np.maximum(p1.min_line - raw, 0.0), np.maximum(raw - p1.cap_line, 0.0))}
    stages["pass1"] = dispatch(sd, p1.st, p1.min_line, p1.cap_line)
    p2 = Pass2(sd, p1.st, p1.min_line, p1.cap_line, ring=ring).run()
    stages["pass2"] = dispatch(sd, p1.st, p1.min_line, p1.cap_line)
    thermal_ok = np.zeros(sd.T, bool)
    thermal_ok[[c for _, c, _, _, _, tag in p1.st.pairs if tag == "p1-thermal"]] = True
    p3 = None
    if sd.config.thermal_valley_charging:
        p3 = Pass3(sd, p1.st, p1.min_line, p1.cap_line).run()
        stages["pass3"] = dispatch(sd, p1.st, p1.min_line, p1.cap_line)
        thermal_ok |= p3.thermal_charge > 1e-6
    checks = validate(sd, p1.st, p1.min_line, p1.cap_line, p1.guarantee_mask, ring, thermal_ok)
    return Result("gtds-commit-ring" if ring else "gtds-commit", p1, p2, stages, checks, iters, p3)
