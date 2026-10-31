"""读取 GTDS "8760h 电力平衡结果汇总表" 与配套机组参数表，转换为算法输入目录。

平衡表每行一个项目、每列一个小时（列标 MMDDHH，如 10100 = 1 月 1 日 0 时）。取：
- 负荷："1. 原始负荷"
- 风电、光伏：出力 + 紧随其后的"（调峰弃电）"，即可发出力；光热按固定出力
- 水电："四、水电出力合计"（按固定曲线）
机组参数表（GTDS 格式）取火电机组、储能电站、备用参数：
- 火电（电站类型 300）按台数展开，最小出力 = min(技术出力, 1) × 单机容量；
  "机组约束"类型 0 为分月最大出力比例（供暖季降出力）
- 抽蓄（381）、新型储能（385）同类合并：充电功率 = 功率 × 储能比率，
  储能库容为满功率充电时长，储能侧容量 = 充电功率 × 库容 × √η
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

THERMAL, PUMPED, BATTERY = 300, 381, 385


def read_balance(path, system: str) -> tuple[list, list[tuple[str, np.ndarray]]]:
    """返回 (时间列标, [(项目名, 8760 数组)])，按表中顺序。"""
    ws = openpyxl.load_workbook(path, data_only=True)[system]
    rows, hdr, cols = [], None, None
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i == 3:                    # 只取时间列（MMDDHH 整数），表尾的汇总列不要
            cols = [k for k, c in enumerate(r) if k >= 2 and isinstance(c, int) and c >= 10100]
            hdr = [r[k] for k in cols]
        if i < 4 or r[1] is None:
            continue
        v = np.array([float(r[k]) if isinstance(r[k], (int, float)) else 0.0 for k in cols])
        rows.append((str(r[1]).strip(), v))
    return hdr, rows


def _row(rows, label, follow=None):
    """按名称取一行；follow 给出时取该行之后第一个名为 follow 的行。"""
    i = next(k for k, (lab, _) in enumerate(rows) if lab == label)
    if follow is None:
        return rows[i][1]
    return next(v for lab, v in rows[i + 1:] if lab == follow)


def balance_series(rows) -> dict[str, np.ndarray]:
    wind = _row(rows, "1. 风电出力") + _row(rows, "1. 风电出力", "（调峰弃电）")
    solar = _row(rows, "2. 光伏出力") + _row(rows, "2. 光伏出力", "（调峰弃电）")
    csp = _row(rows, "3. 光热出力") + _row(rows, "3. 光热出力", "（调峰弃电）")
    return {"load": _row(rows, "1. 原始负荷"), "wind": wind, "solar": solar, "csp": csp,
            "hydro": _row(rows, "四、水电出力合计")}


def _table(wb, name) -> pd.DataFrame:
    rows = list(wb[name].iter_rows(values_only=True))
    head, out = rows[0], []
    for r in rows[1:]:
        if r[0] is None:
            continue
        d = {}
        for h, v in zip(head, r):
            if h is not None and h not in d:
                d[h] = v
        out.append(d)
    return pd.DataFrame(out)


def convert(balance_xlsx, params_xlsx, out_dir, system: str = "西北",
            priority: str = "size") -> dict:
    """priority：火电开机优先级。参数表没有煤耗差别时，"size" 按单机容量从大到小，
    "flex" 按技术出力从低到高（灵活机组先开）。"""
    _, rows = read_balance(balance_xlsx, system)
    ts = balance_series(rows)

    wb = openpyxl.load_workbook(params_xlsx, data_only=True)
    sysrow = _table(wb, "系统表").set_index("名称").loc[system]
    loadrow = _table(wb, "负荷表").set_index("系统ID").loc[sysrow["系统ID"]]
    plants = _table(wb, "电站表")
    plants = plants[plants["有效性"] == 1].set_index("电站ID")
    units = _table(wb, "机组表")
    units = units[(units["有效性"] == 1) & units["电站ID"].isin(plants.index)]
    feat = _table(wb, "特性表").set_index("机组型号")
    loads = [f"负载{x}%" for x in (50, 60, 70, 80, 90)]

    def model_of(cap):
        """按单机容量对应特性表机型（参数表未指定特性 ID 时的默认假设）。"""
        for lim, name in ((1000, "1000MW煤"), (600, "600MW煤超"), (300, "300MW煤"), (200, "200MW煤"),
                          (125, "125MW煤")):
            if cap >= lim:
                return name
        return "100MW煤"

    cons = _table(wb, "机组约束")
    cons = cons[(cons["有效性"] == 1) & (cons["类型"] == 0)].set_index("机组ID")
    months = [f"M{i}" for i in range(1, 13)]

    th, clipped, storages = [], [], {}
    for _, u in units.iterrows():
        p = plants.loc[u["电站ID"]]
        n, cap = int(u["台数"] or 0), float(u["单机容量"])
        if n <= 0:
            continue
        if p["类型"] == THERMAL:
            tech = float(u["技术出力"])
            if tech > 1:
                clipped.append(f"{u['名称']}（技术出力 {tech:.3f}）")
            ratio = ";".join(f"{float(cons.loc[u['机组ID'], m]):.4f}" for m in months) \
                if u["机组ID"] in cons.index else ""
            score = -cap if priority == "size" else tech
            for k in range(n):
                th.append(dict(name=f"{u['名称']}#{k + 1}", capacity=cap, pmin=round(min(tech, 1.0) * cap, 3),
                               coal=score, must_run=False, must_run_weeks="", max_ratio=ratio,
                               maint_days=float(u.get("检修天数") or 0),
                               heat_coef=round(float(feat.loc[model_of(cap), loads].astype(float).mean()), 4),
                               startup=float(feat.loc[model_of(cap), "启停成本"])))
        elif p["类型"] in (PUMPED, BATTERY):
            key = "抽水蓄能" if p["类型"] == PUMPED else "新型储能"
            eta, cr = float(p["储能效率"]), float(p["储能比率"] or 1.0)
            s = storages.setdefault(key, {"power": 0.0, "energy": 0.0, "eta": eta, "cr": cr})
            s["power"] += cap * n
            s["energy"] += cap * n * cr * float(u["储能库容"]) * eta ** 0.5

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ts).round(3).to_csv(out / "timeseries.csv", index=False)
    pd.DataFrame(th).to_csv(out / "units.csv", index=False)
    pd.DataFrame([dict(name=k, power=v["power"], energy=v["energy"], eta_rt=v["eta"], soc_min=0.0,
                       soc_max=1.0, self_discharge_day=0.0, charge_ratio=v["cr"])
                  for k, v in storages.items()]).to_csv(out / "storages.csv", index=False)
    cfg = {"load_reserve_rate": float(loadrow["负荷备用"]),
           "accident_reserve_rate": float(loadrow["事故旋转"]),
           "accident_largest_unit": False,
           "min_reserve": float(loadrow["最小备用"]),
           "min_online_ratio": float(sysrow["保安开机"] or 0),
           "maint_max_ratio": float(sysrow["火电检修max"] or 0.7),
           "coal_price": float(sysrow["标煤单价"] or 700),
           "hydro_in_curtail_base": True}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"系统": system, "负荷TWh": ts["load"].sum() / 1e6, "风电TWh": ts["wind"].sum() / 1e6,
            "光伏TWh": ts["solar"].sum() / 1e6, "光热TWh": ts["csp"].sum() / 1e6,
            "水电TWh": ts["hydro"].sum() / 1e6, "火电台数": len(th),
            "火电容量MW": sum(u["capacity"] for u in th), "储能": storages, "配置": cfg,
            "技术出力大于1已截为1": clipped}


def gtds_storage(rows) -> dict[str, np.ndarray]:
    """平衡表中的储能逐时出力（放电为正、充电为负）。"""
    return {"抽水蓄能": _row(rows, "1. 抽水蓄能发电"), "新型储能": _row(rows, "2. 新型储能发电")}
