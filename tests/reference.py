"""独立参考实现：<=16 交易时枚举全部 2^n 子集求闭合最优。

本模块不导入 ancestorpool 的任何核心算法，只使用被测数据结构
（Transaction 仅作为输入载体）；期望值由这里独立计算，避免“用被测代码
给被测代码作证”。
"""

from __future__ import annotations

from ancestorpool import Transaction


def brute_force_optimum(
    txs: list[Transaction], block_weight: int
) -> tuple[tuple[str, ...], int, int]:
    """枚举所有子集，返回 (入选 txid 拓扑序元组, 总费, 总重)。

    同分按“入选 txid 升序元组”字典序最小破同分；输出顺序单独做拓扑排序。
    """
    n = len(txs)
    idx = {t.txid: i for i, t in enumerate(txs)}
    deps = [{idx[d] for d in t.depends_on} for t in txs]
    w = [t.weight for t in txs]
    f = [t.fee for t in txs]

    best_ids: tuple[str, ...] | None = None
    best_fee = -1
    best_weight = 0

    for mask in range(1 << n):
        members = {i for i in range(n) if (mask >> i) & 1}
        # 依赖闭合
        ok = True
        total_w = 0
        total_f = 0
        for i in members:
            if not deps[i] <= members:
                ok = False
                break
            total_w += w[i]
            total_f += f[i]
        if not ok or total_w > block_weight:
            continue
        ids = tuple(sorted(txs[i].txid for i in members))
        if total_f > best_fee or (total_f == best_fee and (
            best_ids is None or ids < best_ids
        )):
            best_fee = total_f
            best_ids = ids
            best_weight = total_w

    assert best_ids is not None  # 空集恒可行
    id_set = set(best_ids)
    # 拓扑序输出：Kahn（同层 txid 升序）。
    children: dict[str, set[str]] = {t.txid: set() for t in txs}
    indeg: dict[str, int] = {}
    for t in txs:
        if t.txid not in id_set:
            continue
        d = {x for x in t.depends_on if x in id_set}
        indeg[t.txid] = len(d)
        for p in d:
            children[p].add(t.txid)
    ready = sorted(x for x, d in indeg.items() if d == 0)
    order: list[str] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in sorted(children[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                ready.sort()
    return tuple(order), best_fee, best_weight
