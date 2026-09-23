"""求解器测试：与独立穷举参考对拍（n<=16）+ 边界与预算测试。"""

from __future__ import annotations

import math
import random
import unittest

from ancestorpool import (
    DEFAULT_NODE_BUDGET,
    Outpoint,
    SearchBudgetExhaustedError,
    Transaction,
)
from ancestorpool.solver import (
    EXCL_ANCESTOR_EXCLUDED,
    EXCL_CLOSURE_WEIGHT,
    EXCL_NOT_SELECTED,
    solve,
)

from reference import brute_force_optimum
from util import make_tx, random_dag


class BruteForceCrossCheck(unittest.TestCase):
    """对拍：被测求解器必须与独立穷举参考在同输入上完全一致。"""

    def test_handcrafted_cases(self) -> None:
        cases: list[list[Transaction]] = [
            [],
            [make_tx("a", 10, 1)],
            [make_tx("a", 10, 1), make_tx("b", 10, 1, ["a"])],
            # 低费父 + 高费子
            [make_tx("p", 100, 1), make_tx("c", 100, 1000, ["p"])],
            # 菱形依赖
            [
                make_tx("r", 10, 5),
                make_tx("x", 10, 5, ["r"]),
                make_tx("y", 10, 5, ["r"]),
                make_tx("z", 10, 50, ["x", "y"]),
            ],
            # 独立交易（无依赖）
            [make_tx(f"i{i}", 3, i) for i in range(1, 9)],
        ]
        for txs in cases:
            for bw in (1, 5, 20, 50, 10_000):
                with self.subTest(n=len(txs), bw=bw):
                    got = solve(txs, bw)
                    exp_ids, exp_fee, exp_w = brute_force_optimum(txs, bw)
                    self.assertEqual(got.selected, exp_ids)
                    self.assertEqual(got.total_fee, exp_fee)
                    self.assertEqual(got.total_weight, exp_w)
                    self.assertTrue(got.proven_optimal)

    def test_random_dags_against_bruteforce(self) -> None:
        rng = random.Random(20260923)
        for trial in range(120):
            n = rng.randint(0, 12)
            txs = random_dag(rng, n, p_edge=rng.choice([0.0, 0.2, 0.5, 0.8]))
            total_w = sum(t.weight for t in txs)
            if total_w:
                bw = rng.choice(
                    [1, total_w // 4, total_w // 2, total_w, total_w * 2]
                )
            else:
                bw = rng.randint(1, 100)
            got = solve(txs, bw)
            exp_ids, exp_fee, exp_w = brute_force_optimum(txs, bw)
            self.assertEqual(
                (got.selected, got.total_fee, got.total_weight),
                (exp_ids, exp_fee, exp_w),
                msg=f"trial={trial} n={n} bw={bw}",
            )

    def test_size_16_exhaustive_reference_limit(self) -> None:
        # 题面要求的参考上限：16 交易随机 DAG 仍可与 2^16 枚举对拍。
        rng = random.Random(424242)
        for trial in range(6):
            txs = random_dag(rng, 16, p_edge=0.15)
            total_w = sum(t.weight for t in txs)
            bw = total_w // 3
            got = solve(txs, bw)
            exp_ids, exp_fee, _ = brute_force_optimum(txs, bw)
            self.assertEqual(got.selected, exp_ids)
            self.assertEqual(got.total_fee, exp_fee)

    def test_tie_break_lexicographic(self) -> None:
        # 两个独立交易同分同重，只能选一个：字典序选 a。
        txs = [make_tx("b", 10, 7), make_tx("a", 10, 7)]
        r = solve(txs, 10)
        self.assertEqual(r.selected, ("a",))
        # 更复杂同分：容量恰好两件，所有两两组合同费，字典序最小为 {a,b}。
        txs2 = [
            make_tx("a", 10, 5),
            make_tx("z", 10, 5),
            make_tx("b", 10, 5),
            make_tx("c", 10, 5),
        ]
        r2 = solve(txs2, 20)
        self.assertEqual(r2.selected, ("a", "b"))

    def test_output_is_topological(self) -> None:
        rng = random.Random(7)
        for _ in range(50):
            txs = random_dag(rng, rng.randint(1, 14), p_edge=0.4)
            r = solve(txs, sum(t.weight for t in txs))
            pos = {t: i for i, t in enumerate(r.selected)}
            for t in txs:
                if t.txid in pos:
                    for d in t.depends_on:
                        self.assertIn(d, pos)
                        self.assertLess(pos[d], pos[t.txid])


class ExclusionReasonsTest(unittest.TestCase):
    def test_closure_overweight_reason(self) -> None:
        txs = [make_tx("p", 100, 1), make_tx("c", 100, 1000, ["p"])]
        r = solve(txs, 150)  # 闭包重 200 > 150，子不可选；父 100 可选
        self.assertEqual(r.selected, ("p",))
        reasons = {e.txid: e.reason for e in r.exclusions}
        self.assertEqual(reasons["c"], EXCL_CLOSURE_WEIGHT)

    def test_ancestor_excluded_reason(self) -> None:
        # 独立高费 q 占满预算，低费父 p 落选；子 c 的闭包本身不超重，
        # 但因祖先 p 不在包中而不可选 -> ancestor_excluded。
        txs = [
            make_tx("p", 50, 1),
            make_tx("c", 1, 0, ["p"]),
            make_tx("q", 100, 1000),
        ]
        r = solve(txs, 100)
        self.assertEqual(r.selected, ("q",))
        reasons = {e.txid: e.reason for e in r.exclusions}
        self.assertEqual(reasons["c"], EXCL_ANCESTOR_EXCLUDED)

    def test_not_selected_reason(self) -> None:
        txs = [make_tx("a", 100, 1), make_tx("b", 100, 1)]
        r = solve(txs, 150)  # 独立、同费，选字典序小的
        self.assertEqual(r.selected, ("a",))
        reasons = {e.txid: e.reason for e in r.exclusions}
        self.assertEqual(reasons["b"], EXCL_NOT_SELECTED)

    def test_every_non_member_explained(self) -> None:
        rng = random.Random(99)
        txs = random_dag(rng, 12, p_edge=0.3)
        r = solve(txs, sum(t.weight for t in txs) // 2)
        excluded = {e.txid for e in r.exclusions}
        self.assertEqual(excluded, {t.txid for t in txs} - set(r.selected))


class ForcedExclusionTest(unittest.TestCase):
    def test_forced_excluded_matches_independent_enumeration(self) -> None:
        # 独立枚举（不调用被测核心）。
        rng = random.Random(77)
        for trial in range(60):
            n = rng.randint(1, 12)
            txs = random_dag(rng, n, p_edge=rng.choice([0.0, 0.3, 0.6]))
            bw = max(1, sum(t.weight for t in txs) // 2)
            k = rng.randint(1, max(1, n))
            forbidden = set(rng.sample(range(n), k))
            forced = {j: ("missing_input_outpoint", "x") for j in forbidden}
            got = solve(txs, bw, forced_excluded=forced)

            # 独立枚举（不调用被测核心）。
            idx = {t.txid: i for i, t in enumerate(txs)}
            deps = [{idx[d] for d in t.depends_on} for t in txs]
            best: tuple[str, ...] | None = None
            best_fee = -1
            for mask in range(1 << n):
                members = {i for i in range(n) if (mask >> i) & 1}
                if members & forbidden:
                    continue
                tw = tf = 0
                ok = True
                for i in members:
                    if not deps[i] <= members:
                        ok = False
                        break
                    tw += txs[i].weight
                    tf += txs[i].fee
                if not ok or tw > bw:
                    continue
                ids = tuple(sorted(txs[i].txid for i in members))
                if tf > best_fee or (tf == best_fee and (best is None or ids < best)):
                    best_fee, best = tf, ids
            self.assertEqual(
                set(got.selected), set(best or ()), msg=f"trial={trial}"
            )
            self.assertEqual(got.total_fee, best_fee)
            # 每个被强制排除的根都必须带指定理由。
            root_ids = {txs[j].txid for j in forbidden}
            reasons = {e.txid: e.reason for e in got.exclusions}
            self.assertTrue(root_ids <= set(reasons))
            self.assertTrue(all(v == "missing_input_outpoint" for v in reasons.values() if v == "missing_input_outpoint"))


class BudgetTest(unittest.TestCase):
    def test_budget_exhausted_raises_distinct_from_infeasible(self) -> None:
        # 30 个独立交易 + 极小预算，必然在证明最优前耗尽。
        txs = [make_tx(f"t{i:02d}", 1 + i, 100 + i) for i in range(30)]
        with self.assertRaises(SearchBudgetExhaustedError) as cm:
            solve(txs, 10_000, node_budget=3)
        self.assertGreaterEqual(cm.exception.nodes_visited, 3)
        # 空集恒可行：同一输入给足预算必须正常返回（证明有解），不得报“无解”。
        r = solve(txs, 10_000, node_budget=DEFAULT_NODE_BUDGET)
        self.assertTrue(r.proven_optimal)

    def test_budget_must_be_positive_int(self) -> None:
        txs = [make_tx("a", 1, 1)]
        with self.assertRaises(ValueError):
            solve(txs, 10, node_budget=0)
        with self.assertRaises(ValueError):
            solve(txs, 10, node_budget=True)  # bool 拒绝


class InputValidationTest(unittest.TestCase):
    def test_reject_nan_infinity(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                Transaction("t", 1, bad)
            with self.assertRaises(ValueError):
                Transaction("t", bad, 1)
            with self.assertRaises(ValueError):
                Outpoint("x", bad)

    def test_reject_bool_for_int_fields(self) -> None:
        with self.assertRaises(ValueError):
            Transaction("t", True, 1)
        with self.assertRaises(ValueError):
            Transaction("t", 1, False)
        with self.assertRaises(ValueError):
            Outpoint("x", True)

    def test_reject_float_and_negative(self) -> None:
        with self.assertRaises(ValueError):
            Transaction("t", 1.5, 1)
        with self.assertRaises(ValueError):
            Transaction("t", 0, 1)
        with self.assertRaises(ValueError):
            Transaction("t", 1, -1)

    def test_block_weight_validation(self) -> None:
        with self.assertRaises(ValueError):
            solve([], 0)
        with self.assertRaises(ValueError):
            solve([], math.nan)
        with self.assertRaises(ValueError):
            solve([], True)

    def test_dag_validation(self) -> None:
        with self.assertRaises(ValueError):  # 未知依赖
            solve([make_tx("a", 1, 1, ["ghost"])], 100)
        with self.assertRaises(ValueError):  # 重复 id
            solve([make_tx("a", 1, 1), make_tx("a", 1, 1)], 100)

    def test_error_is_locatable(self) -> None:
        try:
            Transaction("t", -5, 1)
        except Exception as e:  # noqa: BLE001
            self.assertIn("weight", str(e))


if __name__ == "__main__":
    unittest.main()
