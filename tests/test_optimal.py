import random
import unittest

from ancestor_pool import (
    AncestorPool,
    make_tx,
    select_optimal,
    select_package,
    OptimalResult,
    ValidationError,
    MissingDependencyError,
    STATUS_OPTIMAL,
    STATUS_BUDGET_EXHAUSTED,
    MAX_CANDIDATES,
)
from tests._reference_optimal import brute_force_optimal_v2


def _pool(txs):
    pool = AncestorPool()
    pool.add_many(txs)
    return pool


class TestOptimalBasic(unittest.TestCase):
    def test_empty_pool(self):
        r = select_optimal(AncestorPool(), 100, node_budget=100)
        self.assertEqual(r.selected, ())
        self.assertEqual(r.total_fee, 0)
        self.assertEqual(r.total_weight, 0)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(r.fee_upper_bound, 0)

    def test_empty_iterable(self):
        r = select_optimal([], 100, node_budget=100)
        self.assertEqual(r.selected, ())
        self.assertEqual(r.status, STATUS_OPTIMAL)

    def test_invalid_arguments(self):
        pool = _pool([make_tx("a", 10, 1, inputs=("x:0",))])
        with self.assertRaises(ValidationError):
            select_optimal(pool, 0)
        with self.assertRaises(ValidationError):
            select_optimal(pool, True)
        with self.assertRaises(ValidationError):
            select_optimal(pool, 10.5)
        with self.assertRaises(ValidationError):
            select_optimal(pool, 100, node_budget=0)
        with self.assertRaises(ValidationError):
            select_optimal(pool, 100, node_budget=False)
        with self.assertRaises(ValidationError):
            select_optimal(42, 100)
        with self.assertRaises(ValidationError):
            select_optimal("not-txs", 100)

    def test_candidate_capacity(self):
        txs = [
            make_tx(f"t{i:02d}", 1, 1, inputs=(f"c{i}:0",))
            for i in range(MAX_CANDIDATES + 1)
        ]
        with self.assertRaises(ValidationError):
            select_optimal(txs, 1000)
        # exactly at the cap is accepted
        r = select_optimal(txs[:-1], 1000, node_budget=10**9)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(len(r.selected), MAX_CANDIDATES)

    def test_duplicate_candidate_ids_rejected(self):
        txs = [
            make_tx("a", 1, 1, inputs=("c0:0",)),
            make_tx("a", 1, 1, inputs=("c1:0",)),
        ]
        with self.assertRaises(ValidationError):
            select_optimal(txs, 10)

    def test_missing_dependency_rejected(self):
        txs = [make_tx("a", 1, 1, inputs=("c:0",), depends=("ghost",))]
        with self.assertRaises(MissingDependencyError):
            select_optimal(txs, 10)

    def test_invalid_tx_field_locatable(self):
        with self.assertRaises(ValidationError) as ctx:
            select_optimal([make_tx("a", 1, 1), object()], 10)
        self.assertIn("candidates[1]", str(ctx.exception))


class TestOptimalSemantics(unittest.TestCase):
    def test_low_fee_ancestor_high_fee_child(self):
        # Greedy by fee rate would take H first and then be forced to
        # add P; the optimum over the cap must be computed exactly.
        txs = [
            make_tx("p", 10, 1, inputs=("c:0",)),      # low-fee parent
            make_tx("h", 10, 100, inputs=("p:0",)),   # high-fee child
            make_tx("k", 15, 90, inputs=("d:0",)),    # rival
        ]
        r = select_optimal(txs, 20, node_budget=10_000)
        # {p,h} fee 101 weight 20 beats {k} fee 90.
        self.assertEqual(list(r.selected), ["p", "h"])
        self.assertEqual(r.total_fee, 101)
        self.assertEqual(r.total_weight, 20)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(r.fee_upper_bound, 101)

    def test_shared_ancestor(self):
        # Two high-fee children share one low-fee ancestor; taking both
        # pays the ancestor only once.
        txs = [
            make_tx("root", 10, 1, inputs=("c:0",)),
            make_tx("c1", 10, 60, inputs=("root:0",)),
            make_tx("c2", 10, 60, inputs=("root:1",)),
            make_tx("solo", 25, 100, inputs=("d:0",)),
        ]
        r = select_optimal(txs, 30, node_budget=10_000)
        # {root,c1,c2} fee 121 weight 30 beats {solo} fee 100.
        self.assertEqual(list(r.selected), ["root", "c1", "c2"])
        self.assertEqual(r.total_fee, 121)

    def test_double_spenders_cannot_coexist(self):
        # A and B spend the same outpoint: mutually exclusive even
        # though both would fit under the weight cap.
        txs = [
            make_tx("a", 10, 50, inputs=("coin:0",)),
            make_tx("b", 10, 40, inputs=("coin:0",)),
        ]
        r = select_optimal(txs, 100, node_budget=10_000)
        self.assertEqual(list(r.selected), ["a"])
        self.assertEqual(r.total_fee, 50)

    def test_double_spend_with_descendant_conflict(self):
        # Conflict beats ancestry: the child's branch loses when the
        # rival of its parent is the better spend.
        txs = [
            make_tx("p", 10, 1, inputs=("coin:0",)),
            make_tx("h", 10, 60, inputs=("p:0",)),
            make_tx("r", 10, 70, inputs=("coin:0",)),  # rival of p
        ]
        r = select_optimal(txs, 100, node_budget=10_000)
        # {p,h} fee 61 vs {r} fee 70; p and r cannot coexist.
        self.assertEqual(list(r.selected), ["r"])
        self.assertEqual(r.total_fee, 70)

    def test_tie_break_smaller_weight_first(self):
        # Equal fee: the lighter package wins regardless of id order.
        txs = [
            make_tx("a", 20, 10, inputs=("x:0",)),
            make_tx("b", 10, 10, inputs=("y:0",)),
        ]
        r = select_optimal(txs, 25, node_budget=10_000)
        self.assertEqual(list(r.selected), ["b"])
        self.assertEqual(r.total_weight, 10)

    def test_tie_break_canonical_id_set(self):
        # Equal fee and equal weight: smallest ascending-id tuple wins.
        txs = [
            make_tx("b", 10, 10, inputs=("x:0",)),
            make_tx("a", 10, 10, inputs=("y:0",)),
        ]
        r = select_optimal(txs, 15, node_budget=10_000)
        self.assertEqual(list(r.selected), ["a"])

    def test_differs_from_old_selector_on_tie(self):
        # Same pool, same cap: the old selector keeps its documented
        # id-tuple tie-break, the new one prefers smaller weight.
        pool = _pool([
            make_tx("a", 20, 10, inputs=("x:0",)),
            make_tx("b", 10, 10, inputs=("y:0",)),
        ])
        old = select_package(pool, 25, node_budget=10_000)
        new = select_optimal(pool, 25, node_budget=10_000)
        self.assertEqual(list(old.selected), ["a"])   # id tuple wins
        self.assertEqual(list(new.selected), ["b"])   # weight wins
        self.assertEqual(old.total_fee, new.total_fee)

    def test_pool_is_read_only(self):
        pool = _pool([
            make_tx("p", 10, 1, inputs=("c:0",)),
            make_tx("h", 10, 9, inputs=("p:0",)),
        ])
        before = (sorted(pool.ids()), str(sorted(pool.ids())))
        r = select_optimal(pool, 25, node_budget=10_000)
        self.assertEqual(sorted(pool.ids()), before[0])
        self.assertIn("p", pool)
        self.assertIn("h", pool)
        self.assertEqual(list(r.selected), ["p", "h"])

    def test_selected_is_topological(self):
        txs = [
            make_tx("z_root", 5, 5, inputs=("c:0",)),
            make_tx("a_child", 5, 5, inputs=("z_root:0",)),
            make_tx("m_grand", 5, 5, inputs=("a_child:0",)),
        ]
        r = select_optimal(txs, 15, node_budget=10_000)
        sel = list(r.selected)
        self.assertLess(sel.index("z_root"), sel.index("a_child"))
        self.assertLess(sel.index("a_child"), sel.index("m_grand"))


class TestOptimalBudget(unittest.TestCase):
    def _pool(self):
        return _pool([
            make_tx("p", 10, 1, inputs=("c:0",)),
            make_tx("h", 10, 9, inputs=("p:0",)),
            make_tx("k", 10, 8, inputs=("d:0",)),
            make_tx("q", 10, 7, inputs=("e:0",)),
        ])

    def test_budget_exhausted_is_not_optimal(self):
        r = select_optimal(self._pool(), 25, node_budget=1)
        self.assertEqual(r.status, STATUS_BUDGET_EXHAUSTED)
        self.assertNotEqual(r.status, STATUS_OPTIMAL)
        self.assertFalse(r.proven_optimal)
        self.assertLessEqual(r.nodes_used, 1)
        # The returned package is still feasible.
        self.assertLessEqual(r.total_weight, 25)
        # The upper bound is admissible: optimum here is 15 ({k,q}).
        self.assertGreaterEqual(r.fee_upper_bound, 15)
        self.assertGreaterEqual(r.fee_upper_bound, r.total_fee)

    def test_generous_budget_proves_optimum(self):
        r = select_optimal(self._pool(), 25, node_budget=10**9)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(r.total_fee, 15)
        self.assertEqual(list(r.selected), ["k", "q"])
        self.assertEqual(r.fee_upper_bound, r.total_fee)


class TestOptimalCrossCheck(unittest.TestCase):
    def test_against_exhaustive_reference(self):
        # Deterministic random candidate sets *with* double spends,
        # cross-checked against the independent 2**n reference.
        rng = random.Random(20260926)
        for trial in range(40):
            n = rng.randrange(1, 11)
            txs = []
            for i in range(n):
                tid = f"t{i:02d}"
                inputs = []
                depends = []
                # ~1/3 of txs double-spend an earlier external coin to
                # create conflicts the pool itself could never hold.
                if i and rng.random() < 0.35:
                    j = rng.randrange(i)
                    inputs.append(f"ext{j}:0")
                else:
                    inputs.append(f"ext{i}:0")
                if i and rng.random() < 0.5:
                    p = rng.randrange(i)
                    inputs.append(f"t{p:02d}:{i}")
                    depends.append(f"t{p:02d}")
                txs.append(
                    make_tx(
                        tid,
                        rng.randrange(1, 21),
                        rng.randrange(0, 31),
                        inputs=inputs,
                        depends=tuple(depends),
                    )
                )
            cap = rng.randrange(5, 61)
            exp_fee, exp_weight, exp_ids = brute_force_optimal_v2(txs, cap)
            r = select_optimal(txs, cap, node_budget=10**9)
            self.assertEqual(r.status, STATUS_OPTIMAL, f"trial {trial}")
            self.assertEqual(r.total_fee, exp_fee, f"trial {trial}")
            self.assertEqual(r.total_weight, exp_weight, f"trial {trial}")
            self.assertEqual(sorted(r.selected), exp_ids, f"trial {trial}")


if __name__ == "__main__":
    unittest.main()
