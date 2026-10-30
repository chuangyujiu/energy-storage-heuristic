"""输出指标与结果文件。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .model import SystemData
from .scheduler import Result

STAGE_NAMES = {"before": "储能投入前", "pass1": "第一遍后", "pass2": "第二遍后", "pass3": "第三遍后"}


def summary(sd: SystemData, res: Result) -> dict:
    ren = float(sd.renewable.sum())
    out = {"方案": res.mode, "环形年迭代次数": res.ring_iters}
    for key, name in STAGE_NAMES.items():
        if key not in res.stages:
            continue
        gen, curt, short = res.stages[key]
        out[name] = {"弃电量_MWh": float(curt.sum()),
                     "弃电率": float(curt.sum() / ren) if ren > 0 else 0.0,
                     "火电发电量_MWh": float(gen.sum()),
                     "缺电量_MWh": float(short.sum())}
    weeks = res.p1.weeks
    out["容量效益"] = {"全年少开台周数": int(sum(w.k for w in weeks)),
                   "平均每周少开台数": float(np.mean([w.k for w in weeks])),
                   "平均每周少开容量_MW": float(np.mean([w.k_mw for w in weeks])),
                   "保供周数": int(sum(w.guarantee for w in weeks))}
    st = res.st
    out["储能"] = {s.name: {"类别": "日储能" if s.kind == "daily" else "周储能",
                          "充电量_MWh": float(st.pch[j].sum()),
                          "放电量_MWh": float(st.pdis[j].sum()),
                          "第一遍放电量_MWh": float(sum(p[4] for p in st.pairs if p[0] == j)),
                          "等效循环次数": float(st.pdis[j].sum() / s.usable) if s.usable else 0.0}
                 for j, s in enumerate(sd.storages)}
    out["校验"] = {k: v["ok"] for k, v in res.checks.items()}
    out["校验全部通过"] = all(v["ok"] for v in res.checks.values())
    return out


def write(sd: SystemData, res: Result, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(w) for w in res.p1.weeks]).rename(columns={
        "week": "周", "start": "起始小时", "end": "结束小时", "N": "初始开机台数",
        "k0": "初始目标k0", "k": "少开台数k", "k_mw": "少开容量MW", "guarantee": "保供周",
        "shortage": "缺电量MWh", "trials": "试算次数"}).assign(周=lambda d: d["周"] + 1) \
        .to_csv(out / "weekly.csv", index=False, encoding="utf-8-sig")

    st, p1 = res.st, res.p1
    hourly = {"小时": np.arange(sd.T), "负荷": sd.load, "净负荷": sd.net_load,
              "容量线": p1.cap_line, "最小出力线": p1.min_line}
    for key, name in STAGE_NAMES.items():
        if key not in res.stages:
            continue
        gen, curt, short = res.stages[key]
        hourly[f"火电_{name}"] = gen
        hourly[f"弃电_{name}"] = curt
        hourly[f"缺电_{name}"] = short
    for j, s in enumerate(sd.storages):
        hourly[f"{s.name}_充电"] = st.pch[j]
        hourly[f"{s.name}_放电"] = st.pdis[j]
        hourly[f"{s.name}_SOC"] = s.energy * s.soc_min + st.lift[j]
    pd.DataFrame(hourly).to_csv(out / "hourly.csv", index=False, encoding="utf-8-sig")

    names = [s.name for s in sd.storages]
    pd.DataFrame([(names[j], c, d, ec, ed, tag) for j, c, d, ec, ed, tag in st.pairs],
                 columns=["储能", "充电小时", "放电小时", "充电电量MWh", "放电电量MWh", "来源"]) \
        .to_csv(out / "pairs_pass1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res.p2.transfers).to_csv(out / "transfers_pass2.csv", index=False,
                                          encoding="utf-8-sig")

    summ = summary(sd, res)
    summ["校验明细"] = res.checks
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    return summ


def print_summary(s: dict):
    print(f"== 方案：{s['方案']}" + (f"（环形年迭代 {s['环形年迭代次数']} 次）" if s["方案"] == "ring" else ""))
    for name in STAGE_NAMES.values():
        if name not in s:
            continue
        v = s[name]
        print(f"  {name:<6} 弃电 {v['弃电量_MWh']:>12,.0f} MWh  弃电率 {v['弃电率']:6.2%}  "
              f"火电 {v['火电发电量_MWh']:>14,.0f} MWh  缺电 {v['缺电量_MWh']:>10,.0f} MWh")
    c = s["容量效益"]
    print(f"  容量效益：平均每周少开 {c['平均每周少开台数']:.2f} 台 / {c['平均每周少开容量_MW']:.0f} MW，"
          f"保供周 {c['保供周数']} 个")
    for name, v in s["储能"].items():
        print(f"  {name}（{v['类别']}）：充电 {v['充电量_MWh']:,.0f} MWh，放电 {v['放电量_MWh']:,.0f} MWh，"
              f"等效循环 {v['等效循环次数']:.0f} 次")
    bad = [k for k, ok in s["校验"].items() if not ok]
    print("  校验：" + ("全部通过" if not bad else "未通过 → " + "、".join(bad)))
