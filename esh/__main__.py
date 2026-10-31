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


def _figures(res_dir: Path):
    """输出典型日电力平衡示意图到 结果目录/figures/（需要 matplotlib）。"""
    try:
        from .plot import plot, typical_days
    except ImportError:
        return
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  （未安装 matplotlib，跳过出图）")
        return
    for name, day in typical_days(res_dir).items():
        plot(res_dir, res_dir / "figures" / f"{name}_{day}.png", day, title_prefix=f"{name} ")
    print(f"  示意图：{res_dir / 'figures'}")


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
    r.add_argument("--no-plot", action="store_true", help="不输出电力平衡示意图")
    r.add_argument("--commit-hours", type=int, help="火电开机锁定周期（小时），如 24 = 按天，默认 168 = 按周")
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
    pl = sub.add_parser("plot", help="画电力平衡结果示意图（可与 GTDS 平衡表对比）")
    pl.add_argument("result", nargs="?", help="run 的结果目录（含 hourly.csv）")
    pl.add_argument("--start", default="01-01", help="起始日期 MM-DD")
    pl.add_argument("--days", type=int, default=1)
    pl.add_argument("--stage", help="阶段：第一遍后 / 第二遍后 / 第三遍后（默认取最后一遍）")
    pl.add_argument("--balance", help="GTDS 8760h 电力平衡结果汇总表，给出时上下对比")
    pl.add_argument("--system", default="西北")
    pl.add_argument("-o", "--out", default="balance.png")
    lb = sub.add_parser("lp-bound", help="第二遍完全前瞻上限（线性规划）对照，需要 scipy")
    lb.add_argument("data", help="输入数据目录")
    lb.add_argument("--commit-hours", type=int, help="火电开机锁定周期（小时）")
    lb.add_argument("--objective", choices=["absorb", "displace"], default="displace",
                    help="absorb=最大化吸收弃电；displace=最大化替代火电（储能放电）")
    fc = sub.add_parser("gtds-commit", help="用 GTDS 平衡表反推的火电开机方案作为输入，只做储能调度")
    fc.add_argument("data", help="输入数据目录（balance 命令生成）")
    fc.add_argument("--balance", required=True, help="GTDS 8760h 电力平衡结果汇总表")
    fc.add_argument("--params", required=True, help="机组参数表（GTDS 格式）")
    fc.add_argument("--system", default="西北")
    fc.add_argument("--no-thermal-reserve", action="store_true", help="火电不承担备用（容量线 = 在运可用容量）")
    fc.add_argument("--valley", action="store_true", help="加第三遍")
    fc.add_argument("-o", "--out", default="results_gtds_commit")
    a = ap.parse_args(argv)

    if a.cmd == "gtds-commit":
        from .fixedcommit import infer_commitment, solve_fixed
        tab, avail, pmn, stat = infer_commitment(a.balance, a.params, a.system)
        print(json.dumps(stat, ensure_ascii=False))
        sd = SystemData.from_dir(a.data)
        sd.config.thermal_valley_charging = a.valley
        if a.no_thermal_reserve:
            sd.config.load_reserve_rate = sd.config.accident_reserve_rate = sd.config.min_reserve = 0
        res = solve_fixed(sd, avail, pmn)
        out = Path(a.out)
        s = report.write(sd, res, out)
        tab.to_csv(out / "gtds_commitment.csv", index=False, encoding="utf-8-sig")
        report.print_summary(s)
        return

    if a.cmd == "lp-bound":
        from .lpbound import compare as lp_compare
        sd = SystemData.from_dir(a.data)
        if a.commit_hours:
            sd.config.commit_hours = a.commit_hours
        r = lp_compare(sd, a.objective)
        k = ("启发式放电_MWh", "LP放电_MWh") if a.objective == "displace" else ("启发式吸收_MWh", "LP上限吸收_MWh")
        h, l = r[k[0]], r[k[1]]
        print(f"第二遍{'替代火电' if a.objective == 'displace' else '吸收弃电'}：启发式 {h / 1e6:.2f} TWh，"
              f"LP 上限 {l / 1e6:.2f} TWh，差距 {(l - h) / 1e6:.2f} TWh（{(l - h) / max(l, 1):.1%}）")
        for name, v in r["分储能"].items():
            print(f"  {name}：启发式 充 {v['启发式充电'] / 1e6:.2f} / 放 {v['启发式放电'] / 1e6:.2f}，"
                  f"LP 充 {v['LP充电'] / 1e6:.2f} / 放 {v['LP放电'] / 1e6:.2f} TWh")
        return

    if a.cmd == "plot":
        from .plot import plot
        print(plot(a.result, a.out, a.start, a.days, a.stage, a.balance, a.system))
        return

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
    if a.commit_hours:
        sd.config.commit_hours = a.commit_hours
    modes = ["base", "ring"] if a.mode == "both" else [a.mode]
    summaries = {}
    for m in modes:
        t0 = time.time()
        res = solve(sd, m)
        s = report.write(sd, res, Path(a.out) / m)
        report.print_summary(s)
        if not a.no_plot:
            _figures(Path(a.out) / m)
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
