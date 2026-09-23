"""固定输入演示：祖先包约束的交易池替换与费用最优包选择。

运行：python demo.py
全程使用仓库内确定性构造的合成数据，无网络、无随机种子外输入、无预录结果；
除正常计算结果外，还会真实触发一个替换拒绝（池原子不变）和一个搜索预算
耗尽异常（与“证明无解”严格区分）。
"""

from __future__ import annotations

from ancestorpool import (
    Outpoint,
    ReplacementRejectedError,
    SearchBudgetExhaustedError,
    Transaction,
    TxPool,
)


def tx(txid: str, weight: int, fee: int, deps=(), inputs=()) -> Transaction:
    return Transaction(
        txid=txid,
        weight=weight,
        fee=fee,
        depends_on=tuple(deps),
        inputs=tuple(Outpoint(t, v) for t, v in inputs),
    )


def hr(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ------------------------------------------------------------------
    # 场景一：一个含链、菱形、独立交易的合成 DAG（固定输入）。
    #   root(低费父) -> child(高费子)：体现“祖先包”必须带上低费父。
    #   diamond：d_top -> {d_l, d_r} -> d_join。
    #   solo*：独立交易。
    # ------------------------------------------------------------------
    hr("场景一：在块重约束内求手续费最优的依赖闭合包")
    pool = TxPool()
    catalog = [
        tx("root", 400, 10),                 # 低费父
        tx("child", 300, 900, deps=["root"]),  # 高费子
        tx("d_top", 200, 20),
        tx("d_l", 200, 30, deps=["d_top"]),
        tx("d_r", 200, 30, deps=["d_top"]),
        tx("d_join", 250, 400, deps=["d_l", "d_r"]),
        tx("soloA", 350, 250),
        tx("soloB", 300, 120),
        tx("heavy", 1600, 9000),           # 单件即超块重：闭包超重演示
    ]
    for t in catalog:
        pool.add(t)

    block_weight = 1400
    result = pool.select(block_weight)
    print(f"块重上限 block_weight = {block_weight}")
    print(f"池内交易数 = {pool.size()}")
    print(f"最优包（拓扑序）= {list(result.selected)}")
    print(f"总手续费 = {result.total_fee}，总权重 = {result.total_weight}")
    print(f"搜索节点数 = {result.nodes_visited}，已证明最优 = {result.proven_optimal}")
    print("每个未入选交易的排除理由：")
    for e in result.exclusions:
        print(f"  - {e.txid}: [{e.reason}] {e.detail}")

    # 校验入选结果确实闭合（演示侧的健全性断言，非核心算法）。
    by_id = {t.txid: t for t in catalog}
    chosen = set(result.selected)
    for tid in chosen:
        for dep in by_id[tid].depends_on:
            assert dep in chosen, f"闭合不变量被破坏: {tid} 缺祖先 {dep}"
    assert result.total_weight <= block_weight

    # ------------------------------------------------------------------
    # 场景二：双花冲突替换（级联移除后代）与替换后孤儿。
    # ------------------------------------------------------------------
    hr("场景二：冲突替换（移除冲突交易及其全部后代）")
    p2 = TxPool()
    p2.add(tx("alpha", 400, 40, inputs=[("external", 0)]))
    p2.add(tx("beta", 300, 20, deps=["alpha"]))          # alpha 的后代
    p2.add(tx("gamma", 300, 20, deps=["beta"]))          # 间接后代
    p2.add(tx("delta", 200, 500, inputs=[("alpha", 0)]))  # 无显式依赖
    print("替换前池内:", [t.txid for t in p2.transactions()])

    newcomer = tx("replacement", 500, 200, inputs=[("external", 0)])
    add_res = p2.add(newcomer)
    print(f"新交易 replacement: fee={newcomer.fee}, weight={newcomer.weight}")
    print(f"被移除集合（冲突者+全部后代）= {list(add_res.removed)}")
    print("替换后池内:", [t.txid for t in p2.transactions()])
    print("替换后孤儿（outpoint 创建者已移除）:", p2.orphans())
    sel2 = p2.select(5000)
    print("替换后可选包:", list(sel2.selected))
    for e in sel2.exclusions:
        print(f"  - {e.txid}: [{e.reason}] {e.detail}")

    # ------------------------------------------------------------------
    # 边界一（真实触发的拒绝）：费率不严格更高 -> 整体拒绝，池原子不变。
    # ------------------------------------------------------------------
    hr("边界一：替换被拒绝（费用/费率严格条件不满足），池保持原子不变")
    before = [t.txid for t in p2.transactions()]
    # replacement 费率 200/500 = 0.40。令新交易费用 201 > 200（费用条件通过），
    # 但费率 201/1000 = 0.201 < 0.40（费率条件不通过）-> 仍被拒绝。
    weak = tx("weak_repl", 1000, 201, inputs=[("external", 0)])
    try:
        p2.add(weak)
    except ReplacementRejectedError as err:
        print(f"拒绝原因代码 = {err.reason}")
        print(f"错误信息 = {err}")
    after = [t.txid for t in p2.transactions()]
    print(f"池在拒绝前后完全一致: {before == after}（{after}）")

    # 再演示“费用够但费率等号不通过”的严格性。
    p3 = TxPool()
    p3.add(tx("x", 100, 10, inputs=[("e", 0)]))
    try:
        # 11/110 == 10/100：费用更高但费率相等，必须拒绝。
        p3.add(tx("y", 110, 11, inputs=[("e", 0)]))
    except ReplacementRejectedError as err:
        print(f"等号费率拒绝原因代码 = {err.reason}（严格大于才通过）")

    # ------------------------------------------------------------------
    # 边界二（真实触发的异常）：搜索预算耗尽，区别于“证明无解”。
    # ------------------------------------------------------------------
    hr("边界二：搜索预算耗尽（与‘证明无解’严格区分）")
    many = [
        tx(f"m{i:02d}", 10 + (i * 37 % 53), 100 + (i * 61 % 200))
        for i in range(30)
    ]
    print(f"构造 {len(many)} 笔互相独立的交易，给极小节点预算 node_budget=5")
    try:
        from ancestorpool.solver import solve
        solve(many, 10_000, node_budget=5)
    except SearchBudgetExhaustedError as err:
        print(f"捕获 {type(err).__name__}: {err}")
        print("含义：搜索没做完，最优性无法证明 —— 并不代表无可行解。")
    # 同一输入给足预算：空集恒可行，且能证明最优，对照“有解可证”。
    full = TxPool()
    for t in many:
        full.add(t)
    ok = full.select(10_000)
    print(
        f"给足预算后正常返回：入选 {len(ok.selected)} 笔，总费 {ok.total_fee}，"
        f"已证明最优 = {ok.proven_optimal}"
    )

    print()
    print("演示完成：以上费用/费率/排除理由均为现场真实计算结果。")


if __name__ == "__main__":
    main()
