"""与 GTDS 储能出力（发+/充− 电力一览表）对比。"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

KINDS = {"抽水蓄能": "抽水蓄能", "新型储能": "新型储能"}


def read_gtds_storage(path) -> dict[str, np.ndarray]:
    """按储能类别汇总 GTDS 逐时出力（放电为正、充电为负）。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    out: dict[str, np.ndarray] = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        title = str(rows[0][1])
        kind = next((v for k, v in KINDS.items() if k in title), None)
        if kind is None:
            continue
        cols = [i for i, c in enumerate(rows[3]) if isinstance(c, str) and re.fullmatch(r"\d+时", c.strip())]
        s = np.array([[float(r[i] or 0) for i in cols] for r in rows
                      if isinstance(r[0], str) and re.match(r"\d{4}/", r[0].strip())]).ravel()
        out[kind] = out.get(kind, 0) + s
    return out


def _stats(x: np.ndarray, net_load: np.ndarray, curt0: np.ndarray, month, hour) -> dict:
    dis, ch = np.maximum(x, 0), np.maximum(-x, 0)
    days = x.reshape(-1, 24)
    return {
        "年放电量_GWh": dis.sum() / 1e3,
        "年充电量_GWh": ch.sum() / 1e3,
        "放电小时数": int((dis > 1e-6).sum()),
        "充电小时数": int((ch > 1e-6).sum()),
        "平均放电功率_MW": dis.sum() / max((dis > 1e-6).sum(), 1),
        "充电电量中无弃电时段占比": float((ch * (curt0 <= 1)).sum() / max(ch.sum(), 1e-9)),
        "充电电量中净负荷为正时段占比": float((ch * (net_load > 0)).sum() / max(ch.sum(), 1e-9)),
        "同日既充又放天数": int(((days > 1e-6).any(1) & (days < -1e-6).any(1)).sum()),
        "_month": pd.Series(dis).groupby(month).sum() / 1e3,
        "_hour": pd.Series(x).groupby(hour).mean() / 1e3,
    }


def compare(gtds_xlsx, result_dir, out_dir=None, gtds_series: dict | None = None) -> dict:
    """gtds_series 给出时直接使用（如 8760h 平衡表中的储能出力），不读 gtds_xlsx。"""
    res = Path(result_dir)
    h = pd.read_csv(res / "hourly.csv")
    T = len(h)
    month = (pd.Timestamp("2030-01-01") + pd.to_timedelta(np.arange(T), "h")).month
    hour = np.arange(T) % 24
    nl, curt0 = h["净负荷"].to_numpy(), h["弃电_储能投入前"].to_numpy()
    g = gtds_series if gtds_series is not None else read_gtds_storage(gtds_xlsx)

    rows, monthly, hourly = [], {}, {}
    for kind, gx in g.items():
        mine = (h[f"{kind}_放电"] - h[f"{kind}_充电"]).to_numpy()
        a, b = _stats(gx[:T], nl, curt0, month, hour), _stats(mine, nl, curt0, month, hour)
        corr = float(np.corrcoef(gx[:T], mine)[0, 1])
        dcorr = float(np.corrcoef(gx[:T].reshape(-1, 24).sum(1), mine.reshape(-1, 24).sum(1))[0, 1])
        for k in a:
            if not k.startswith("_"):
                rows.append({"储能": kind, "指标": k, "GTDS": a[k], "本算法": b[k]})
        rows.append({"储能": kind, "指标": "逐时相关系数", "GTDS": None, "本算法": corr})
        rows.append({"储能": kind, "指标": "日净放电量相关系数", "GTDS": None, "本算法": dcorr})
        monthly[kind] = pd.DataFrame({"GTDS": a["_month"], "本算法": b["_month"]})
        hourly[kind] = pd.DataFrame({"GTDS": a["_hour"], "本算法": b["_hour"]})

    table = pd.DataFrame(rows)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "compare_summary.csv", index=False, encoding="utf-8-sig")
        pd.concat(monthly, axis=1).to_csv(out / "compare_monthly_GWh.csv", encoding="utf-8-sig")
        pd.concat(hourly, axis=1).to_csv(out / "compare_hourly_profile_GW.csv", encoding="utf-8-sig")
    return {"summary": table, "monthly": monthly, "hourly": hourly}
