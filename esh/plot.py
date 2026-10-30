"""电力平衡结果示意图（仿 GTDS"电力平衡结果示意图"）。

逐时堆叠：火电（其中储能充电部分以"储能负荷"砖纹显示）→ 水电 → 光热 → 核电/外来电 →
储能发电 → 光伏 → 风电 → 弃光 → 弃风 → 弃水 → 电力不足；虚线为原始负荷。
有储能充电时，堆叠顶 = 负荷 + 储能充电 + 弃电。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = [  # (列名, 图例名, 填充色, 纹理)
    ("火电", "火力发电", "#E8302A", None),
    ("储能负荷", "储能负荷", "#1C2FB8", "++"),
    ("水电", "水力发电", "#22E3EE", None),
    ("光热", "光热发电", "#F39A1E", "||"),
    ("核电", "核电", "#9C6ADE", None),
    ("外来电", "外来电", "#8C8C8C", None),
    ("储能发电", "储能发电", "#3D7FE6", "xx"),
    ("光伏", "光伏发电", "#F7EA1B", "||"),
    ("风电", "风力发电", "#18C418", "||"),
    ("弃光", "光伏弃光", "#E8DC7A", "///"),
    ("弃风", "风电弃风", "#8FCB5A", "///"),
    ("弃水", "水电弃水", "#9DEFF3", "///"),
    ("缺电", "电力不足", "#FFFFFF", ".."),
]


def _setup_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def frame_from_result(result_dir, stage: str | None = None) -> pd.DataFrame:
    """由 run 的 hourly.csv 得到各层逐时功率。弃电按风、光可发出力比例分摊，超出部分记弃水。"""
    h = pd.read_csv(Path(result_dir) / "hourly.csv")
    if stage is None:
        stage = next(s for s in ("第三遍后", "第二遍后", "第一遍后") if f"火电_{s}" in h)
    col = lambda c: h[c].to_numpy(float) if c in h else np.zeros(len(h))
    ch = sum(col(c) for c in h if c.endswith("_充电"))
    dis = sum(col(c) for c in h if c.endswith("_放电"))
    wind, solar, hydro = col("风电可发"), col("光伏可发"), col("水电")
    curt = col(f"弃电_{stage}")
    ws = wind + solar
    c_ws = np.minimum(curt, ws)
    share = np.divide(wind, ws, out=np.zeros_like(ws), where=ws > 0)
    cw, cs = c_ws * share, c_ws * (1 - share)
    cw_h = np.minimum(curt - c_ws, hydro)
    return pd.DataFrame({
        "负荷": col("负荷"), "火电": col(f"火电_{stage}"), "储能充电": ch, "储能发电": dis,
        "水电": hydro - cw_h, "光热": col("光热"), "核电": col("核电"), "外来电": col("外来电"),
        "光伏": solar - cs, "风电": wind - cw, "弃光": cs, "弃风": cw, "弃水": cw_h,
        "缺电": col(f"缺电_{stage}")})


def frame_from_gtds(balance_xlsx, system: str = "西北") -> pd.DataFrame:
    """由 GTDS 8760h 电力平衡结果汇总表得到同样的各层逐时功率。"""
    from .balance import _row, read_balance
    _, rows = read_balance(balance_xlsx, system)
    st = _row(rows, "1. 抽水蓄能发电") + _row(rows, "2. 新型储能发电")
    g = lambda lab, f=None: _row(rows, lab, f)
    return pd.DataFrame({
        "负荷": g("1. 原始负荷"), "火电": g("六、火电出力合计"),
        "储能充电": np.maximum(-st, 0), "储能发电": np.maximum(st, 0),
        "水电": g("四、水电出力合计"), "光热": g("3. 光热出力"), "核电": 0.0, "外来电": 0.0,
        "光伏": g("2. 光伏出力"), "风电": g("1. 风电出力"),
        "弃光": g("2. 光伏出力", "（调峰弃电）"), "弃风": g("1. 风电出力", "（调峰弃电）"),
        "弃水": g("四、水电出力合计", "（调峰弃电）"), "缺电": g("二、系统电力不足")})


def _window(start: str, days: int) -> slice:
    t0 = pd.Timestamp(f"2030-{start}" if start.count("-") == 1 else start)
    a = int((t0 - pd.Timestamp(t0.year, 1, 1)).total_seconds() // 3600)
    return slice(a, a + 24 * days)


def draw(ax, df: pd.DataFrame, w: slice, title: str, ymax: float | None = None):
    d = df.iloc[w].reset_index(drop=True)
    x = np.arange(len(d))
    thermal, charge = d["火电"].to_numpy(float), d["储能充电"].to_numpy(float)
    layers = d.copy()
    layers["火电"] = np.maximum(thermal - charge, 0)      # 储能充电部分画在火电顶部（砖纹）
    layers["储能负荷"] = charge
    bottom = np.zeros(len(d))
    for key, label, color, hatch in LAYERS:
        y = layers[key].to_numpy(float)
        if key != "火电" and not (y > 1e-6).any():
            continue
        b = thermal - charge if key == "储能负荷" else bottom
        ax.bar(x, y, width=1.0, bottom=b, align="edge", color=color, hatch=hatch,
               edgecolor="#333333", linewidth=0.3, label=label)
        bottom = thermal if key in ("火电", "储能负荷") else bottom + y
    xs = np.append(x, len(d))
    ax.step(xs, np.append(d["负荷"], d["负荷"].iloc[-1]), where="post", color="#E01BD0",
            linestyle="--", linewidth=1.4, label="原始负荷")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("P / MW")
    ax.set_xlim(0, len(d))
    ax.set_ylim(min(0.0, float((thermal - charge).min()) * 1.05), ymax)
    step = 4 if len(d) <= 48 else 24
    ax.set_xticks(np.arange(0, len(d) + 1, step))
    if len(d) <= 48:
        ax.set_xticklabels([f"{h % 24 if h % 24 or h == 0 else 24}时"
                            for h in np.arange(0, len(d) + 1, step)])
    else:
        days = pd.Timestamp(2030, 1, 1) + pd.to_timedelta(w.start + np.arange(0, len(d) + 1, step), "h")
        ax.set_xticklabels([f"{t.month}/{t.day}" for t in days])
    ax.grid(axis="both", linestyle=":", color="#999999", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    return float(bottom.max())


def typical_days(result_dir, stage=None) -> dict[str, str]:
    """典型日：净负荷最大日、弃电最大日、储能放电最大日（MM-DD）。"""
    h = pd.read_csv(Path(result_dir) / "hourly.csv")
    df = frame_from_result(result_dir, stage)
    daily = lambda s: np.asarray(s, float)[: len(s) // 24 * 24].reshape(-1, 24)
    picks = {"净负荷最大日": daily(h["净负荷"]).max(1).argmax(),
             "弃电最大日": daily(df["弃光"] + df["弃风"] + df["弃水"]).sum(1).argmax(),
             "储能放电最大日": daily(df["储能发电"]).sum(1).argmax()}
    out = {}
    for k, d in picks.items():
        t = pd.Timestamp(2030, 1, 1) + pd.Timedelta(days=int(d))
        out[k] = f"{t.month:02d}-{t.day:02d}"
    return out


def plot(result_dir=None, out="balance.png", start="01-01", days=1, stage=None,
         gtds_balance=None, system="西北", title_prefix=""):
    """画本算法结果；给出 gtds_balance 时上下两幅对比（同一纵轴范围）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()
    w = _window(start, days)
    panels = []
    if gtds_balance:
        panels.append(("GTDS", frame_from_gtds(gtds_balance, system)))
    if result_dir:
        panels.append(("本算法", frame_from_result(result_dir, stage)))
    top = 0.0
    for _, df in panels:
        d = df.iloc[w]
        top = max(top, float((d.drop(columns=["负荷", "储能充电"]).sum(1)).max()),
                  float(d["负荷"].max()))
    ymax = top * 1.08
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 6.2 * len(panels)), squeeze=False)
    t0 = pd.Timestamp(2030, 1, 1) + pd.Timedelta(hours=w.start)
    span = f"{t0.month}月{t0.day}日" + (f" 起 {days} 天" if days > 1 else "")
    for ax, (name, df) in zip(axes[:, 0], panels):
        draw(ax, df, w, f"{title_prefix}{span} 电力平衡结果示意图（{name}）", ymax)
    seen = {}
    for ax in axes[:, 0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / fig.get_figheight()))   # 顶部留给图例
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
