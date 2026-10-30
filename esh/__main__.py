"""命令行入口。

    python -m esh demo examples/demo              # 生成示例数据
    python -m esh run examples/demo -o results    # 基准方案 + 环形年对比
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import report, synthetic
from .model import SystemData
from .scheduler import solve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="esh", description="储能启发式调度算法")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="生成示例输入数据")
    d.add_argument("out", nargs="?", default="examples/demo")
    d.add_argument("--seed", type=int, default=2026)
    r = sub.add_parser("run", help="运行调度算法")
    r.add_argument("data", help="输入数据目录")
    r.add_argument("-o", "--out", default="results")
    r.add_argument("--mode", choices=["base", "ring", "both"], default="both",
                   help="base=年初 SOC 取下限；ring=环形年；both=两种都跑并比较（默认）")
    r.add_argument("--valley", action="store_true", help="第三遍：弃电不足时允许火电在低谷时段充电")
    b = sub.add_parser("balance", help="把 GTDS 8760h 电力平衡表 + 机组参数表转换为输入目录")
    b.add_argument("--balance", required=True, help="8760h 电力平衡结果汇总表")
    b.add_argument("--params", required=True, help="机组参数表（GTDS 格式）")
    b.add_argument("--system", default="西北")
    b.add_argument("--priority", choices=["size", "flex"], default="size",
                   help="火电开机优先级：size=单机容量从大到小；flex=技术出力从低到高")
    b.add_argument("-o", "--out", default="data/balance")
    g = sub.add_parser("gtds", help="把 GTDS 的 Excel 数据转换为输入目录")
    g.add_argument("--units", required=True, help="机组参数表（含系统表、负荷表、电站表、机组表）")
    g.add_argument("--plants", required=True, help="电站出力一览表（G/U 表）")
    g.add_argument("--load-dir", required=True, help="日历负荷 L*-8760-*.xlsx 所在目录")
    g.add_argument("--system", default="西南")
    g.add_argument("--accident-rate", type=float, default=0.04, help="事故备用率（默认取事故旋转备用）")
    g.add_argument("--security", action="store_true", help="计入系统表的保安开机比例")
    g.add_argument("--coal-must-run", action="store_true", help="煤电全年全开（与 GTDS 模拟结果一致）")
    g.add_argument("-o", "--out", default="data/gtds")
    c = sub.add_parser("compare-gtds", help="与 GTDS 储能出力对比")
    c.add_argument("--storage", help="GTDS 储能电站 发+/充- 电力一览表")
    c.add_argument("--balance", help="GTDS 8760h 电力平衡结果汇总表（代替 --storage）")
    c.add_argument("--system", default="西北", help="平衡表中的系统名（配合 --balance）")
    c.add_argument("result", help="run 的结果目录（含 hourly.csv）")
    a = ap.parse_args(argv)

    if a.cmd == "compare-gtds":
        import pandas as pd
        from .compare import compare
        series = None
        if a.balance:
            from .balance import gtds_storage, read_balance
            series = gtds_storage(read_balance(a.balance, a.system)[1])
        r = compare(a.storage, a.result, a.result, series)
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print(r["summary"].round(2).to_string(index=False))
            for k in r["monthly"]:
                print(f"\n[{k}] 月放电量 GWh\n", r["monthly"][k].round(0).T.to_string())
                print(f"[{k}] 逐时平均出力 GW（放+/充−）\n", r["hourly"][k].round(2).T.to_string())
        return

    if a.cmd == "balance":
        from .balance import convert as bconvert
        info = bconvert(a.balance, a.params, a.out, a.system, a.priority)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    if a.cmd == "gtds":
        from .gtds import convert
        info = convert(a.units, a.plants, a.load_dir, a.out, a.system, a.accident_rate,
                       security=a.security, coal_must_run=a.coal_must_run)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    if a.cmd == "demo":
        print(f"示例数据已写入 {synthetic.make(a.out, a.seed)}")
        return

    sd = SystemData.from_dir(a.data)
    if a.valley:
        sd.config.thermal_valley_charging = True
    modes = ["base", "ring"] if a.mode == "both" else [a.mode]
    summaries = {}
    for m in modes:
        t0 = time.time()
        res = solve(sd, m)
        s = report.write(sd, res, Path(a.out) / m)
        report.print_summary(s)
        print(f"  用时 {time.time() - t0:.1f} s，结果目录 {Path(a.out) / m}\n")
        summaries[m] = s
    if len(modes) == 2:
        cmp = {m: {"第二遍后弃电量_MWh": s["第二遍后"]["弃电量_MWh"],
                   "平均每周少开容量_MW": s["容量效益"]["平均每周少开容量_MW"],
                   "缺电量_MWh": s["第二遍后"]["缺电量_MWh"]} for m, s in summaries.items()}
        (Path(a.out) / "compare.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        print("== 方案比较", json.dumps(cmp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
