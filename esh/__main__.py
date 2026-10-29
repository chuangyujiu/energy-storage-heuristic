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
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        print(f"示例数据已写入 {synthetic.make(a.out, a.seed)}")
        return

    sd = SystemData.from_dir(a.data)
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
