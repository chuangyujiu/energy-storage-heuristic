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
    # 跨年电量即年初 SOC：第一遍削峰配对 20，第二遍把 12 月末剩余的 10 MWh 弃电带到年初
    assert sum(p[3] for p in ring.st.pairs if p[1] > p[2]) == pytest.approx(20.0)
    assert ring.st.lift[0, -1] == pytest.approx(30.0)
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


def test_pass3_thermal_valley_charging():
    """无弃电时，火电低谷（中午）充电、高峰（晚上）放电，削峰填谷且不越线。"""
    T = 168
    nl = np.full(T, 300.0)
    for d in range(7):
        nl[d * 24 + 12: d * 24 + 15] = 250.0     # 低谷（6 台最小出力线 240，无弃电）
        nl[d * 24 + 19: d * 24 + 21] = 420.0     # 高峰（6 台容量线 500）
    sd = toy(nl, [Storage("bat", 50, 100, 0.81)])
    sd.config.thermal_valley_charging = True
    res = solve(sd)
    st = res.st
    assert res.stages["pass2"][1].sum() == pytest.approx(0.0, abs=1e-6)
    assert st.pch[0].sum() > 0 and st.pdis[0].sum() > 0
    gen = res.stages["pass3"][0]
    assert gen[19:21].max() < 420.0 and gen[12:15].min() > 250.0
    assert_checks(res)


def test_pass3_skips_when_spread_below_losses():
    """峰谷差小于储能损耗时不做火电充放。"""
    T = 168
    nl = np.full(T, 300.0)
    for d in range(7):
        nl[d * 24 + 12: d * 24 + 15] = 290.0
        nl[d * 24 + 19: d * 24 + 21] = 310.0     # 290 / 310 = 0.935 > η = 0.81
    sd = toy(nl, [Storage("bat", 50, 100, 0.81)])
    sd.config.thermal_valley_charging = True
    res = solve(sd)
    assert res.st.pch[0].sum() == pytest.approx(0.0, abs=1e-6)
    assert_checks(res)


def test_pass3_flat_plateau_before_valley():
    """窗口开头与夜间高峰一样高（平台）时，放电只排在充电之后，仍能削峰填谷。"""
    T = 168
    nl = np.full(T, 420.0)
    for d in range(7):
        nl[d * 24 + 12: d * 24 + 16] = 250.0
    sd = toy(nl, [Storage("bat", 50, 100, 0.81)])
    sd.config.thermal_valley_charging = True
    res = solve(sd)
    assert res.st.pch[0].sum() > 0
    assert_checks(res)


def test_charge_ratio_and_monthly_derate():
    """充电功率 = 功率 × 充电比例；分月最大出力比例折减容量线。"""
    T = 168
    nl = np.full(T, 250.0)
    nl[40] = 320.0
    nl[34:37] = 150.0
    bat = Storage("bat", power=5, energy=100, eta_rt=1.0, charge_ratio=2.0)
    sd = toy(nl, [bat])
    res = solve(sd)
    assert res.st.pch[0].max() <= 10.0 + 1e-9 and res.st.pch[0].max() > 5.0
    assert_checks(res)

    sd2 = toy(np.full(T, 250.0), [])
    for u in sd2.units:
        u.max_ratio = (0.5,) * 12                 # 每台只有 50 MW 可用
    from esh.commitment import Fleet
    f = Fleet(sd2)
    order = f.order(0)
    cap, mn = f.lines(order, 6, np.arange(T))
    assert cap[0] == pytest.approx(6 * 50 - 100)  # 事故备用仍按最大单机 100 MW


def test_daily_commitment_stops_units_on_light_days():
    """按天锁定开机：只有一天有尖峰时，其余各天可以少开机。"""
    nl = np.full(168, 250.0)
    nl[40] = 320.0                       # 第 2 天尖峰 → 当天需 5 台，其余 4 台
    sd = toy(nl, [])
    res = solve(sd)
    assert [w.N for w in res.p1.weeks] == [5]
    sd.config.commit_hours = 24
    res = solve(sd)
    assert [w.N for w in res.p1.weeks] == [4, 5, 4, 4, 4, 4, 4]
    assert_checks(res)


def test_lp_bound_not_below_heuristic():
    """LP 上限不低于启发式第二遍（同一第一遍结果）。"""
    pytest.importorskip("scipy")
    from esh.lpbound import compare
    T = 168
    nl = np.full(T, 500.0)
    nl[10:14] = 250.0
    nl[34:38] = 250.0
    nl[80] = 900.0
    sd = toy(nl, [Storage("s", 100, 200, 0.81)], n_units=12)
    r = compare(sd, "displace")
    assert r["LP放电_MWh"] >= r["启发式放电_MWh"] - 1e-6


def test_fixed_commitment_storage_only():
    """开机方案固定时，只为削峰需求和消纳安排储能，校验通过。"""
    from esh.fixedcommit import solve_fixed
    T = 168
    nl = np.full(T, 300.0)
    for d in range(7):
        nl[d * 24 + 11: d * 24 + 15] = 150.0     # 中午弃电（最小出力线 200）
        nl[d * 24 + 19] = 420.0                  # 晚峰超过可用容量 400
    sd = toy(nl, [Storage("bat", 50, 200, 1.0)])
    sd.config.min_reserve = 0
    res = solve_fixed(sd, np.full(T, 400.0), np.full(T, 200.0))
    peak = sum(p[4] for p in res.st.pairs if p[2] == 19 and p[5] == "p1")
    assert peak == pytest.approx(20.0)                    # 第一遍只削掉超出可用容量的 20
    assert res.stages["pass2"][2].sum() == pytest.approx(0.0, abs=1e-6)
    assert_checks(res)


def test_energy_share_splits_peak_across_storages():
    """按电量比例分担：同一削峰需求由两类储能按可用电量比例同时承担。"""
    sd = toy(_peak_week(), [Storage("a", 50, 100, 1.0), Storage("b", 50, 300, 1.0)])
    sd.config.storage_share = "energy"
    res = solve(sd)
    p1 = {}
    for j, c, d, e, ed, tag in res.st.pairs:
        if d == 40 and tag == "p1":
            p1[j] = p1.get(j, 0) + ed
    assert p1[0] == pytest.approx(5.0) and p1[1] == pytest.approx(15.0)   # 20 按 100:300 分
    assert_checks(res)


def test_pass2_ring_wraps_year_end():
    """环形年第二遍：年末弃电充的电可放到 1 月 1 日凌晨，年末 SOC = 年初 SOC。"""
    T = 168
    nl = np.full(T, 500.0)
    nl[T - 4:] = 250.0              # 年末最后 4 小时弃电（最小出力线 300），之后年内再无放电时段
    st = State(T, 1)
    sd = toy(nl, [Storage("s", 100, 200, 1.0)], n_units=12)
    base = Pass2(sd, State(T, 1), np.full(T, 300.0), np.full(T, 2000.0)).run()
    assert base.st.pch[0].sum() == pytest.approx(0.0, abs=1e-6)     # 不跨年时年末充的电放不出去
    ring = Pass2(sd, st, np.full(T, 300.0), np.full(T, 2000.0), ring=True).run()
    assert st.pch[0].sum() == pytest.approx(200.0)
    assert st.pdis[0, :10].sum() > 0                                 # 放到年初
    assert st.lift[0, -1] == pytest.approx(st.pdis[0].sum())        # 年末存量 = 年初放出的电


def test_dynamic_share_runs_and_validates():
    """按剩余电量动态分担：结果通过全部校验。"""
    sd = toy(_peak_week(), [Storage("a", 50, 100, 1.0), Storage("b", 50, 300, 0.81)])
    sd.config.storage_share = "dynamic"
    res = solve(sd)
    assert res.p1.weeks[0].k == 1
    assert_checks(res)


def test_maintenance_scheduled_in_low_load_period():
    """等备用法：检修安排在备用最充裕（负荷低）的时段，且同时检修容量不超过上限。"""
    from esh.commitment import Fleet
    T = 24 * 20
    nl = np.full(T, 300.0)
    nl[24 * 10:24 * 13] = 50.0           # 第 11～13 天负荷最低
    sd = toy(nl, [])
    sd.config.commit_hours = 24
    sd.config.maintenance = True
    sd.config.maint_max_ratio = 0.2       # 6 × 100 MW 的 20% → 同时最多检修 1 台
    for u in sd.units:
        u.maint_days = 3
    f = Fleet(sd)
    assert f.maint.sum(0).max() <= 1
    assert f.maint[:, 10:13].sum() == 3  # 最低负荷的 3 天里安排了一台
    res = solve(sd)
    assert_checks(res)


def test_startup_aware_order_prefers_units_already_on():
    """计入启停成本：前一周期已开的机组排在未开的同类机组前面；燃料成本低的排前。"""
    from esh.commitment import Fleet
    sd = toy(np.full(48, 250.0), [])
    sd.config.startup_aware = True
    for i, u in enumerate(sd.units):
        u.heat_coef, u.startup = 1.0, 6.0
    sd.units[5].heat_coef = 0.99                      # 燃料最便宜
    f = Fleet(sd)
    assert f.order(0)[0] == 5
    order = f.order(0, prev_on={3, 4})
    assert set(order[:2]) == {3, 4}                   # 已开机组免启动成本，排最前
