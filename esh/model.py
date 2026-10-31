"""输入数据结构：火电机组、储能、系统时序与参数配置。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

HOURS_PER_WEEK = 168
EPS = 1e-6          # 判断"有/无"的数值容差（MW、MWh）
DAILY_WINDOWS = (24, 48, 72)
WEEKLY_WINDOWS = tuple(range(24, 169, 24))


@dataclass
class ThermalUnit:
    name: str
    capacity: float                 # 容量 MW
    pmin: float                     # 最小技术出力 MW
    coal: float                     # 煤耗 g/kWh，决定开机优先级
    must_run: bool = False          # 是否必开
    must_run_weeks: tuple | None = None  # 仅在这些周（0 起）必开，如供暖季；None 表示 must_run 全年有效
    max_ratio: tuple | None = None  # 分月最大出力比例（12 个值，如供暖季 0.8）；None 表示全年为 1

    def is_must_run(self, week: int) -> bool:
        if not self.must_run:
            return False
        return self.must_run_weeks is None or week in self.must_run_weeks


@dataclass
class Storage:
    name: str
    power: float                    # 额定（放电）功率 MW
    energy: float                   # 额定容量 MWh
    eta_rt: float                   # 往返效率
    soc_min: float = 0.0            # SOC 下限（占额定容量比例）
    soc_max: float = 1.0            # SOC 上限
    self_discharge_day: float = 0.0  # 每天自损耗率，日储能取 0
    charge_ratio: float = 1.0       # 充电功率 / 放电功率（如抽蓄 1.1）

    @property
    def p_ch(self) -> float:
        return self.power * self.charge_ratio

    @property
    def duration(self) -> float:
        return self.energy / self.power

    @property
    def kind(self) -> str:
        return "daily" if self.duration < 9 else "weekly"

    @property
    def usable(self) -> float:
        """可用容量 = (SOC 上限 − SOC 下限) × 额定容量。"""
        return self.energy * (self.soc_max - self.soc_min)

    @property
    def eta_c(self) -> float:
        return math.sqrt(self.eta_rt)

    @property
    def eta_d(self) -> float:
        return math.sqrt(self.eta_rt)

    @property
    def rho(self) -> float:
        """每小时保留率 1 − σ_h。"""
        return (1.0 - self.self_discharge_day) ** (1.0 / 24.0)

    @property
    def windows(self) -> tuple:
        return DAILY_WINDOWS if self.kind == "daily" else WEEKLY_WINDOWS


def storage_weights(storages: list[Storage], share: str) -> np.ndarray:
    """按比例同时分担时各储能的份额：energy 按放电侧可用电量，power 按额定放电功率。"""
    w = np.array([s.usable * s.eta_d if share == "energy" else s.power for s in storages], float)
    return w / w.sum() if w.sum() > 0 else w


def storage_order(storages: list[Storage]) -> list[int]:
    """调用顺序：短时优先（日储能 → 周储能），同类内效率优先。"""
    return sorted(range(len(storages)),
                  key=lambda j: (storages[j].kind != "daily", -storages[j].eta_rt))


@dataclass
class Config:
    load_reserve_rate: float = 0.03       # 负荷备用率
    accident_reserve_rate: float = 0.08   # 事故备用率
    ring_max_iter: int = 5                # 环形年最大迭代次数
    hydro_in_curtail_base: bool = False   # 弃电率分母是否包含水电
    min_online_ratio: float = 0.0         # 保安开机：在运火电容量 ≥ 该比例 × 火电总装机
    thermal_valley_charging: bool = False  # 第三遍：弃电不足时允许火电在低谷时段充电
    valley_window_start: int = 10          # 第三遍调度窗口起点（每天该时刻起 24 h）
    valley_ratio: float | None = None      # 低谷水位 ≤ 比例 × 高峰水位才充放；默认取储能往返效率
    accident_largest_unit: bool = True     # 事故备用是否不低于在运最大单机
    min_reserve: float = 0.0               # 最小备用 MW
    commit_hours: int = 168                # 火电开机锁定周期（168 = 按周，24 = 按天）
    storage_share: str = "priority"        # 多类储能分工：priority 短时优先；energy / power 按电量 / 功率比例同时分担


@dataclass
class SystemData:
    load: np.ndarray        # 逐时总负荷 MW（用于备用）
    net_load: np.ndarray    # 净负荷 MW
    renewable: np.ndarray   # 风电+光伏（可选含水电）可发功率 MW，弃电率的分母
    units: list[ThermalUnit]
    storages: list[Storage]
    config: Config = field(default_factory=Config)
    components: dict = field(default_factory=dict)   # 各类电源逐时出力（风电、光伏可发等），用于出图

    @property
    def T(self) -> int:
        return len(self.net_load)

    @property
    def month(self) -> np.ndarray:
        """每小时所在月份（0–11），按非闰年从 1 月 1 日 0 时起算。"""
        days = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24
        return np.minimum(np.searchsorted(days, np.arange(self.T), side="right") - 1, 11)

    def week_bounds(self) -> list[tuple[int, int]]:
        """开机锁定周期的边界。默认按 168 h 分周，最后一周并入余数（8760 h → 52 周，末周 192 h）；
        config.commit_hours = 24 时按天。"""
        P = int(self.config.commit_hours)
        n = max(1, self.T // P)
        b = [(w * P, (w + 1) * P) for w in range(n)]
        b[-1] = (b[-1][0], self.T)
        return b

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_dir(cls, path: str | Path) -> "SystemData":
        """读取目录：timeseries.csv、units.csv、storages.csv、config.json（可选）。

        timeseries.csv 列：load, wind, solar, csp, nuclear, import, hydro（缺失列按 0）。
        净负荷 = 负荷 − 风电 − 光伏 − 光热 − 核电 − 外来电净受入 − 水电。
        """
        path = Path(path)
        ts = pd.read_csv(path / "timeseries.csv")
        col = lambda c: ts[c].to_numpy(float) if c in ts else np.zeros(len(ts))
        load = col("load")
        cfg = Config()
        if (path / "config.json").exists():
            cfg = Config(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        renewable = col("wind") + col("solar") + col("csp") + (col("hydro") if cfg.hydro_in_curtail_base else 0)
        net = (load - col("wind") - col("solar") - col("csp") - col("nuclear") - col("import")
               - col("hydro"))

        units = []
        for r in pd.read_csv(path / "units.csv").to_dict("records"):
            weeks = r.get("must_run_weeks")
            if isinstance(weeks, str) and weeks.strip():
                weeks = tuple(_parse_weeks(weeks))
            else:
                weeks = None
            mr = r.get("max_ratio")
            mr = tuple(float(x) for x in mr.split(";")) if isinstance(mr, str) and mr.strip() else None
            units.append(ThermalUnit(str(r["name"]), float(r["capacity"]), float(r["pmin"]),
                                     float(r["coal"]), bool(r.get("must_run", False)), weeks, mr))

        storages = []
        for r in pd.read_csv(path / "storages.csv").to_dict("records"):
            storages.append(Storage(str(r["name"]), float(r["power"]), float(r["energy"]),
                                    float(r["eta_rt"]), float(r.get("soc_min", 0.0)),
                                    float(r.get("soc_max", 1.0)),
                                    float(r.get("self_discharge_day", 0.0)),
                                    float(r.get("charge_ratio", 1.0) or 1.0)))

        comps = {k: col(k) for k in ("wind", "solar", "csp", "hydro", "nuclear", "import") if k in ts}
        return cls(load, net, renewable, units, storages, cfg, comps)


def _parse_weeks(s: str) -> list[int]:
    """解析 "0-11;45-51" 形式的周列表（周序号从 0 开始）。"""
    out = []
    for part in s.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)
