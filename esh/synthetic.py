"""生成一套示例数据（虚构省级系统，仅用于演示算法，不代表任何真实电网）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HEATING_WEEKS = "0-10;45-51"   # 供暖季：约 11 月中旬至次年 3 月中旬


def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def make(out_dir: str | Path, seed: int = 2026, T: int = 8760) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = np.arange(T)
    hod, doy = h % 24, h // 24

    # 负荷：冬夏双高峰、早晚双峰、周末下降
    season = 1 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365) ** 2 + 0.12 * np.exp(-((doy - 200) / 30) ** 2)
    daily = 0.82 + 0.10 * np.exp(-((hod - 11) / 3) ** 2) + 0.16 * np.exp(-((hod - 19) / 2.5) ** 2) \
        - 0.08 * np.exp(-((hod - 4) / 3) ** 2)
    weekend = np.where((doy % 7) >= 5, 0.94, 1.0)
    load = 9500 * season * daily * weekend * (1 + _ar1(rng, T, 0.95, 0.008))

    # 风电：多日天气过程，春秋大、夏季小，夜间略大
    wcf = 0.27 + 0.08 * np.cos(2 * np.pi * (doy - 80) / 182.5) + 0.03 * np.cos(2 * np.pi * hod / 24)
    wind = 4500 * np.clip(wcf * np.exp(_ar1(rng, T, 0.985, 0.09) - 0.3), 0, 0.95)

    # 光伏：日内钟形 × 季节 × 逐日云量
    elev = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * (hod >= 6) * (hod <= 18)
    cloud = np.repeat(np.clip(rng.beta(4, 1.6, T // 24 + 1), 0.15, 1), 24)[:T]
    solar = 5500 * 0.82 * elev ** 1.3 * (0.8 + 0.2 * np.cos(2 * np.pi * (doy - 172) / 365)) * cloud

    nuclear = np.full(T, 1000.0)
    imp = 1200 + 300 * np.exp(-((hod - 19) / 3) ** 2)
    hydro = 600 + 300 * np.exp(-((doy - 190) / 40) ** 2)

    pd.DataFrame({"load": load, "wind": wind, "solar": solar, "nuclear": nuclear,
                  "import": imp, "hydro": hydro}).round(2).to_csv(out / "timeseries.csv", index=False)

    units = []
    for i in range(4):
        units.append(dict(name=f"超超临界{i + 1}", capacity=1000, pmin=400, coal=275 + i,
                          must_run=False, must_run_weeks=""))
    for i in range(6):
        units.append(dict(name=f"60万{i + 1}", capacity=600, pmin=270, coal=295 + i,
                          must_run=False, must_run_weeks=""))
    for i in range(6):
        units.append(dict(name=f"热电{i + 1}", capacity=350, pmin=175, coal=305 + i,
                          must_run=True, must_run_weeks=HEATING_WEEKS))
    for i in range(4):
        units.append(dict(name=f"30万{i + 1}", capacity=300, pmin=150, coal=320 + i,
                          must_run=False, must_run_weeks=""))
    pd.DataFrame(units).to_csv(out / "units.csv", index=False)

    pd.DataFrame([
        dict(name="电化学储能", power=800, energy=1600, eta_rt=0.85, soc_min=0.1, soc_max=0.9,
             self_discharge_day=0.0),
        dict(name="抽水蓄能", power=1200, energy=8400, eta_rt=0.75, soc_min=0.0, soc_max=1.0,
             self_discharge_day=0.0),
        dict(name="压缩空气储能", power=300, energy=3600, eta_rt=0.70, soc_min=0.05, soc_max=0.95,
             self_discharge_day=0.01),
    ]).to_csv(out / "storages.csv", index=False)

    (out / "config.json").write_text(json.dumps(
        {"load_reserve_rate": 0.03, "accident_reserve_rate": 0.08}, indent=2), encoding="utf-8")
    return out
