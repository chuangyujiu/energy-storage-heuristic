import numpy as np
import pytest

from esh import Config, Storage, SystemData, ThermalUnit, solve
from esh.pass2 import Pass2
from esh.state import State
from esh.waterfill import waterfill


def toy(net_load, storages, n_units=6):
    """6 台 100 MW 机组（最小出力 40 MW），备用率为 0 → 事故备用 = 最大单机 100 MW，
    即开 n 台时容量线 = 100n − 100，最小出力线 = 40n。"""
    nl = np.asarray(net_load, float)
    units = [ThermalUnit(f"G{i}", 100, 40, 300 + i) for i in range(n_units)]
    return SystemData(np.full(nl.size, 1000.0), nl, np.full(nl.size, 100.0), units, storages,
                      Config(load_reserve_rate=0.0, accident_reserve_rate=0.0))


def assert_checks(res):
    bad = {k: v for k, v in res.checks.items() if not v["ok"]}
    assert not bad, bad


# ---------------------------------------------------------------- 灌水法
def test_waterfill_levels_deepest_first():
    v = np.array([10.0, 6.0, 2.0])
    x = waterfill(v, np.full(3, 100.0), np.ones(3), 6.0, floor=0.0)
    # 最深的谷先削到和第二深一样（4），再一起往下削（各 1）
    np.testing.assert_allclose(x, [5.0, 1.0, 0.0])


def test_waterfill_respects_caps_and_weights():
    v = np.array([10.0, 10.0])
    x = waterfill(v, np.array([1.0, 100.0]), np.array([1.0, 0.5]), 3.0, floor=0.0)
    assert x[0] == pytest.approx(1.0)
    assert np.dot(x, [1.0, 0.5]) == pytest.approx(3.0)


def test_waterfill_returns_max_when_target_unreachable():
    x = waterfill(np.array([3.0, 1.0]), np.array([2.0, 5.0]), np.ones(2), 100.0, floor=0.0)
    np.testing.assert_allclose(x, [2.0, 1.0])


# ---------------------------------------------------------------- 第一遍
def _peak_week(T=336):
    nl = np.full(T, 250.0)
    nl[40] = 320.0                 # 第 2 天一个尖峰 → 当天需开 5 台
    nl[34:37] = 150.0              # 尖峰前 4~6 h 的低谷 → 停 1 台后最小出力线 160，弃电 10×3
    return nl


def test_pass1_storage_allows_one_less_unit():
    bat = Storage("bat", power=50, energy=100, eta_rt=1.0)
    res = solve(toy(_peak_week(), [bat]))
    w0 = res.p1.weeks[0]
    assert (w0.N, w0.k) == (5, 1)
    assert w0.k_mw == 100
    p1 = [p for p in res.st.pairs if p[5] == "p1"]
    assert sum(p[4] for p in p1) == pytest.approx(20.0)     # 第一遍削峰 320 − 300
    assert {p[1] for p in p1} == {34, 35, 36} and {p[2] for p in p1} == {40}
    # 第二遍再把剩余弃电 10 吸收掉，放到同一个峰
    assert res.st.pch[0, 34:37].sum() == pytest.approx(30.0)
    assert res.stages["pass2"][1].sum() == pytest.approx(0.0, abs=1e-6)
    assert_checks(res)


def test_pass1_insufficient_curtailment_keeps_unit():
    nl = _peak_week()
    nl[34:37] = 155.0              # 弃电只有 15 < 20
    res = solve(toy(nl, [Storage("bat", 50, 100, 1.0)]))
    assert res.p1.weeks[0].k == 0
    assert_checks(res)


def test_pass1_efficiency_increases_charge_need():
    nl = _peak_week()
    nl[34:37] = 152.0              # 弃电 24：η=1 可行，η=0.8 需要 25 不可行
    assert solve(toy(nl, [Storage("bat", 50, 100, 1.0)])).p1.weeks[0].k == 1
    assert solve(toy(nl, [Storage("bat", 50, 100, 0.8)])).p1.weeks[0].k == 0


def test_pass1_soc_capacity_limits_pairing():
    res = solve(toy(_peak_week(), [Storage("bat", 50, 15, 1.0)]))   # 容量 15 < 20
    assert res.p1.weeks[0].k == 0
    assert_checks(res)


def test_pass1_window_expands_for_weekly_storage():
    nl = _peak_week()
    nl[34:37] = 250.0
    nl[0:3] = 150.0                # 弃电在尖峰前 38~40 h：日储能 48 h 窗口可达
    res = solve(toy(nl, [Storage("bat", 50, 100, 1.0)]))
    assert res.p1.weeks[0].k == 1
    assert_checks(res)


def test_guarantee_mode_records_shortage():
    nl = np.full(336, 300.0)
    nl[50] = 560.0                 # 全部 6 台容量线 500，缺 60
    nl[40:44] = 450.0              # 可用火电充电空间 50×4
    res = solve(toy(nl, [Storage("bat", 30, 100, 1.0)]))
    w0 = res.p1.weeks[0]
    assert w0.guarantee and w0.k == 0
    assert w0.shortage == pytest.approx(30.0)              # 储能放电功率 30，剩余 30 记缺电
    assert res.stages["pass2"][2].sum() == pytest.approx(30.0)
    assert_checks(res)


# ---------------------------------------------------------------- 环形年
def test_ring_year_wraps_december_curtailment():
    T = 336
    nl = np.full(T, 250.0)
    nl[10] = 320.0                 # 第 1 周开头的尖峰，年内往回找不到弃电
    nl[T - 6:T - 3] = 150.0        # 年末 12 月的弃电
    sd = toy(nl, [Storage("bat", 50, 100, 1.0)])
    base, ring = solve(sd, "base"), solve(sd, "ring")
    assert base.p1.weeks[0].k == 0
    assert ring.p1.weeks[0].k == 1
    assert ring.st.lift[0, -1] == pytest.approx(20.0)      # 跨年配对电量即年初 SOC
    assert_checks(base)
    assert_checks(ring)


# ---------------------------------------------------------------- 第二遍
def test_pass2_nearby_pairing_example():
    """文档例子：储能 200 MWh，周一、周二中午各弃电 200，全周最大净负荷在周四。
    就近配对（每天腾空）能吸收 400。"""
    T = 168
    nl = np.full(T, 500.0)
    nl[10:14] = 250.0              # 周一中午：低于最小出力线 300 共 200
    nl[34:38] = 250.0              # 周二中午
    nl[80] = 900.0                 # 周四的峰
    st = State(T, 1)
    sd = toy(nl, [Storage("s", 100, 200, 1.0)])
    p2 = Pass2(sd, st, np.full(T, 300.0), np.full(T, 2000.0)).run()
    assert st.pch[0].sum() == pytest.approx(400.0)
    assert st.pdis[0].sum() == pytest.approx(400.0)
    assert len(p2.transfers) == 2
    # 周一的电在周二弃电开始前放完
    assert st.lift[0, 33] == pytest.approx(0.0, abs=1e-6)
    # 放电不制造新的弃电：火电始终不低于最小出力线
    assert (nl + st.ch_total - st.dis_total >= 300.0 - 1e-6)[~np.isin(np.arange(T), [*range(10, 14), *range(34, 38)])].all()


def test_pass2_peak_shaving_within_interval():
    T = 48
    nl = np.full(T, 500.0)
    nl[2:4] = 250.0                # 弃电 100
    nl[10], nl[11] = 700.0, 650.0
    st = State(T, 1)
    Pass2(toy(nl, [Storage("s", 200, 300, 1.0)]), st, np.full(T, 300.0), np.full(T, 2000.0)).run()
    # 最大峰先削到和第二大一样（650），再一起往下削：700→625、650→625
    assert st.pdis[0, 10] == pytest.approx(75.0)
    assert st.pdis[0, 11] == pytest.approx(25.0)


def test_demo_dataset_validates(tmp_path):
    from esh import synthetic
    sd = SystemData.from_dir(synthetic.make(tmp_path, T=168 * 6))
    res = solve(sd)
    assert_checks(res)
    curt = [res.stages[k][1].sum() for k in ("before", "pass1", "pass2")]
    assert curt[0] >= curt[1] - 1e-6 >= curt[2] - 2e-6


def test_commitment_keeps_reserve_above_min_line():
    """净负荷为负时也要开够机组，使容量线 ≥ 最小出力线（备用留在最小出力之上）。"""
    from esh.commitment import Fleet
    sd = toy(np.full(168, -500.0), [])
    sd.config.load_reserve_rate = 0.2          # 负荷 1000 × 20% + 最大单机 100 = 300
    f = Fleet(sd)
    order = f.order(0)
    n = f.required(order, np.arange(168), 0)
    cap, mn = f.lines(order, n, np.arange(168))
    assert n == 5 and (cap >= mn).all()        # 100n − 300 ≥ 40n → n ≥ 5


def test_security_commitment_floor():
    """保安开机 50%：6 × 100 MW 中至少开 3 台，即使负荷只需 2 台。"""
    sd = toy(np.full(168, 50.0), [])
    sd.config.min_online_ratio = 0.5
    res = solve(sd)
    w = res.p1.weeks[0]
    assert w.N == 3 and w.k == 0
    assert_checks(res)
