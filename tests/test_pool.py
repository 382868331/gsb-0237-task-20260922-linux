"""TxPool 替换语义测试：冲突、后代移除、费用/费率严格性、原子性、孤儿、间接冲突。"""

from __future__ import annotations

import unittest

from ancestorpool import (
    Outpoint,
    PoolFullError,
    ReplacementRejectedError,
    Transaction,
    TxPool,
)
from ancestorpool.pool import (
    REJ_DEPENDS_ON_REMOVED,
    REJ_FEE_NOT_GREATER,
    REJ_FEERATE_NOT_HIGHER,
)

from util import make_tx

OP = Outpoint


class AddBasicsTest(unittest.TestCase):
    def test_add_and_order_independent_snapshot(self) -> None:
        p = TxPool()
        p.add(make_tx("b", 1, 1))
        p.add(make_tx("a", 1, 1))
        self.assertEqual([t.txid for t in p.transactions()], ["a", "b"])
        self.assertEqual(p.size(), 2)

    def test_unknown_dependency_rejected(self) -> None:
        p = TxPool()
        with self.assertRaises(Exception):
            p.add(make_tx("x", 1, 1, ["ghost"]))
        self.assertEqual(p.size(), 0)

    def test_duplicate_id_in_pool_rejected(self) -> None:
        p = TxPool()
        p.add(make_tx("a", 1, 1))
        with self.assertRaises(ReplacementRejectedError) as cm:
            p.add(make_tx("a", 1, 2))
        self.assertEqual(cm.exception.reason, "already_in_pool")
        self.assertEqual(p.get("a").fee, 1)  # 未被覆盖


class ReplacementPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = TxPool()
        # A 消耗外部 outpoint E；B 依赖 A；C 依赖 B（三代链）。
        self.p.add(make_tx("A", 100, 10, inputs=[("E", 0)]))
        self.p.add(make_tx("B", 100, 10, deps=["A"]))
        self.p.add(make_tx("C", 100, 10, deps=["B"]))

    def test_conflict_removes_conflict_and_all_descendants(self) -> None:
        # N 双花 E，费用 31 > 30，费率 31/100 > 30/300：移除 A,B,C。
        r = self.p.add(make_tx("N", 100, 31, inputs=[("E", 0)]))
        self.assertEqual(r.removed, ("A", "B", "C"))
        self.assertEqual(
            {t.txid for t in self.p.transactions()}, {"N"}
        )

    def test_indirect_conflict_via_descendant_chain(self) -> None:
        # 只双花 E，但 B、C 是 A 的间接/直接后代，必须一起移除。
        r = self.p.add(make_tx("N", 50, 100, inputs=[("E", 0)]))
        self.assertIn("A", r.removed)
        self.assertIn("B", r.removed)
        self.assertIn("C", r.removed)

    def test_fee_must_be_strictly_greater(self) -> None:
        # 被移除集合总费 30；30 不满足“严格大于”。
        with self.assertRaises(ReplacementRejectedError) as cm:
            self.p.add(make_tx("N", 100, 30, inputs=[("E", 0)]))
        self.assertEqual(cm.exception.reason, REJ_FEE_NOT_GREATER)
        self.assertEqual(self.p.size(), 3)  # 原子不变

    def test_feerate_must_be_strictly_higher(self) -> None:
        # 新费 31 > 30，但费率 31/1000 < 30/300：拒绝。
        with self.assertRaises(ReplacementRejectedError) as cm:
            self.p.add(make_tx("N", 1000, 31, inputs=[("E", 0)]))
        self.assertEqual(cm.exception.reason, REJ_FEERATE_NOT_HIGHER)
        self.assertEqual(self.p.size(), 3)

    def test_equal_feerate_rejected(self) -> None:
        # 31/310 == 30/300：等号不通过（必须严格更高）。
        with self.assertRaises(ReplacementRejectedError) as cm:
            self.p.add(make_tx("N", 310, 31, inputs=[("E", 0)]))
        self.assertEqual(cm.exception.reason, REJ_FEERATE_NOT_HIGHER)

    def test_higher_feerate_but_lower_fee_rejected(self) -> None:
        # N 费率更高（29/100 < 30/300? 否；选 29/50=0.58 > 0.1）但费 29 < 30。
        with self.assertRaises(ReplacementRejectedError) as cm:
            self.p.add(make_tx("N", 50, 29, inputs=[("E", 0)]))
        self.assertEqual(cm.exception.reason, REJ_FEE_NOT_GREATER)

    def test_new_tx_must_not_depend_on_removed(self) -> None:
        with self.assertRaises(ReplacementRejectedError) as cm:
            self.p.add(
                make_tx("N", 100, 100, deps=["A"], inputs=[("E", 0)])
            )
        self.assertEqual(cm.exception.reason, REJ_DEPENDS_ON_REMOVED)
        self.assertEqual(self.p.size(), 3)

    def test_new_tx_must_not_spend_removed_output(self) -> None:
        # N 双花 E 的同时还消耗 A:0（隐式依赖被移除者）：拒绝。
        with self.assertRaises(ReplacementRejectedError) as cm:
            self.p.add(
                make_tx("N", 100, 100, inputs=[("E", 0), ("A", 0)])
            )
        self.assertEqual(cm.exception.reason, REJ_DEPENDS_ON_REMOVED)


class RejectionAtomicityTest(unittest.TestCase):
    def test_failed_add_leaves_pool_unchanged(self) -> None:
        p = TxPool()
        p.add(make_tx("A", 100, 10, inputs=[("E", 0)]))
        p.add(make_tx("B", 100, 10, deps=["A"]))
        before = tuple(t.txid for t in p.transactions())
        for bad in (
            make_tx("N", 1000, 31, inputs=[("E", 0)]),   # 费率不足
            make_tx("N", 1, 1, deps=["ghost"]),            # 未知依赖
            make_tx("A", 1, 1),                            # id 已存在
        ):
            with self.assertRaises(Exception):
                p.add(bad)
            self.assertEqual(
                tuple(t.txid for t in p.transactions()), before
            )


class OrphanAfterReplacementTest(unittest.TestCase):
    def test_descendant_without_explicit_dep_becomes_orphan(self) -> None:
        p = TxPool()
        p.add(make_tx("A", 100, 10, inputs=[("E", 0)]))
        # D 消耗 A 的输出但没有 depends_on（替换后将成孤儿）。
        p.add(make_tx("D", 100, 5, inputs=[("A", 0)]))
        r = p.add(make_tx("N", 100, 50, inputs=[("E", 0)]))
        self.assertEqual(r.removed, ("A",))
        orphans = p.orphans()
        self.assertIn("D", orphans)
        self.assertIn("A:0", orphans["D"])
        # select 强制排除孤儿并给出理由。
        sel = p.select(10_000)
        self.assertEqual(sel.selected, ("N",))
        reasons = {e.txid: e.reason for e in sel.exclusions}
        self.assertEqual(reasons["D"], "missing_input_outpoint")

    def test_orphan_descendant_excluded_from_package(self) -> None:
        p = TxPool()
        p.add(make_tx("A", 100, 10, inputs=[("E", 0)]))
        p.add(make_tx("D", 100, 5, inputs=[("A", 0)]))
        # G 显式依赖 D；D 成孤儿后 G 因祖先被排除而不可选。
        p.add(make_tx("G", 100, 500, deps=["D"]))
        p.add(make_tx("N", 100, 50, inputs=[("E", 0)]))
        sel = p.select(10_000)
        self.assertEqual(sel.selected, ("N",))
        reasons = {e.txid: e.reason for e in sel.exclusions}
        self.assertEqual(reasons["D"], "missing_input_outpoint")
        self.assertEqual(reasons["G"], "ancestor_excluded")


class PoolLimitTest(unittest.TestCase):
    def test_pool_full_on_plain_add(self) -> None:
        p = TxPool()
        for i in range(30):
            p.add(make_tx(f"t{i:02d}", 1, 1))
        with self.assertRaises(PoolFullError):
            p.add(make_tx("one_more", 1, 1))

    def test_replacement_allowed_at_limit(self) -> None:
        p = TxPool()
        for i in range(29):
            p.add(make_tx(f"t{i:02d}", 1, 1))
        p.add(make_tx("victim", 100, 10, inputs=[("E", 0)]))
        self.assertEqual(p.size(), 30)
        # 替换 victim（移除 1 加入 1），即使在满员时也允许。
        r = p.add(make_tx("repl", 100, 11, inputs=[("E", 0)]))
        self.assertEqual(r.removed, ("victim",))
        self.assertEqual(p.size(), 30)


class LowFeeParentHighFeeChildTest(unittest.TestCase):
    def test_package_requires_parent_but_optimizes(self) -> None:
        p = TxPool()
        p.add(make_tx("parent", 100, 1))           # 低费父
        p.add(make_tx("child", 100, 1000, ["parent"]))  # 高费子
        # 容量只够父：拿不到高费子。
        self.assertEqual(p.select(100).selected, ("parent",))
        # 容量够闭包：父子同选，费用 1001。
        r = p.select(200)
        self.assertEqual(r.selected, ("parent", "child"))
        self.assertEqual(r.total_fee, 1001)
        # 容量 150：闭包 200 超重，只能选父。
        self.assertEqual(p.select(150).selected, ("parent",))


if __name__ == "__main__":
    unittest.main()
