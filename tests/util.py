"""测试辅助：确定性合成 DAG 生成（与被测核心算法完全独立）。"""

from __future__ import annotations

import random

from ancestorpool import Outpoint, Transaction


def make_tx(txid: str, weight: int, fee: int, deps=(), inputs=()) -> Transaction:
    return Transaction(
        txid=txid,
        weight=weight,
        fee=fee,
        depends_on=tuple(deps),
        inputs=tuple(Outpoint(tx, v) for tx, v in inputs),
    )


def random_dag(
    rng: random.Random,
    n: int,
    *,
    p_edge: float = 0.35,
    max_weight: int = 400,
    max_fee: int = 300,
    input_prob: float = 0.7,
) -> list[Transaction]:
    """生成 n 笔交易的随机 DAG（只允许向更早的交易连依赖边），保证无环。

    每个交易以 input_prob 概率消耗 1 个外部 outpoint（ext:i:0），用于双花冲突
    测试时由调用方自行安排；这里的外部 outpoint 互不相同，不产生冲突。
    """
    txs: list[Transaction] = []
    ids = [f"t{i:02d}" for i in range(n)]
    for i in range(n):
        deps = sorted(
            {ids[j] for j in range(i) if rng.random() < p_edge}
        )
        inputs = []
        if rng.random() < input_prob:
            inputs.append((f"ext{i}", 0))
        txs.append(
            Transaction(
                txid=ids[i],
                weight=rng.randint(1, max_weight),
                fee=rng.randint(0, max_fee),
                depends_on=deps,
                inputs=tuple(Outpoint(*o) for o in inputs),
            )
        )
    return txs
