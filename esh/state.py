"""储能状态：每种储能一套长度为 T 的 P_ch、P_dis 与 SOC 抬高量数组。

lift[j, t] = SOC[j, t] − SOC 下限，即第 t 小时末尚未放出的配对电量之和。
年初 SOC 取下限，因此基准方案中 lift 初值为 0。
"""
from __future__ import annotations

import numpy as np


class State:
    def __init__(self, T: int, n_storage: int):
        self.T = T
        self.pch = np.zeros((n_storage, T))
        self.pdis = np.zeros((n_storage, T))
        self.lift = np.zeros((n_storage, T))
        self.ch_total = np.zeros(T)
        self.dis_total = np.zeros(T)
        self.shortage = np.zeros(T)
        self.pairs: list[tuple] = []    # (储能, 充电小时, 放电小时, 充电电量, 放电电量, 来源)

    def snapshot(self):
        return (self.pch.copy(), self.pdis.copy(), self.lift.copy(), self.ch_total.copy(),
                self.dis_total.copy(), self.shortage.copy(), list(self.pairs))

    def restore(self, snap):
        (self.pch[:], self.pdis[:], self.lift[:], self.ch_total[:], self.dis_total[:],
         self.shortage[:]) = snap[:6]
        self.pairs[:] = snap[6]
