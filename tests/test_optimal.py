import unittest

from ancestor_pool import (
    AncestorPool,
    make_tx,
    select_optimal,
    select_package,
    MissingDependencyError,
    ValidationError,
    STATUS_OPTIMAL,
    STATUS_BUDGET_EXHAUSTED,
    MAX_CANDIDATES,
)
from tests._reference_optimal import brute_force_conflict_optimal
from tests._data import gen_txs, gen_txs_with_conflicts


def _build_pool(txs):
    pool = AncestorPool()
    pool.add_many(txs)
    return pool


class TestOptimalBasic(unittest.TestCase):
    def test_empty_pool(self):
        r = select_optimal(AncestorPool(), 100, node_budget=10)
        self.assertEqual(r.selected, ())
        self.assertEqual(r.total_fee, 0)
        self.assertEqual(r.total_weight, 0)
        self.assertEqual(r.fee_upper_bound, 0)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertTrue(r.proven_optimal)

    def test_empty_candidate_list(self):
        r = select_optimal([], 100, node_budget=10)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(r.ledger, ())

    def test_invalid_arguments(self):
        pool = _build_pool([make_tx("a", 10, 1, inputs=("x:0",))])
        with self.assertRaises(ValidationError):
            select_optimal(pool, 0, node_budget=10)
        with self.assertRaises(ValidationError):
            select_optimal(pool, True, node_budget=10)
        with self.assertRaises(ValidationError):
            select_optimal(pool, -5, node_budget=10)
        with self.assertRaises(ValidationError):
            select_optimal(pool, 100, node_budget=0)
        with self.assertRaises(ValidationError):
            select_optimal(pool, 100, node_budget=False)
        with self.assertRaises(ValidationError):
            select_optimal("not-a-pool", 100, node_budget=10)
        with self.assertRaises(ValidationError):
            select_optimal(42, 100, node_budget=10)

    def test_candidate_list_validation_errors(self):
        # duplicate ids
        dup = [make_tx("a", 10, 1, inputs=("x:0",)),
               make_tx("a", 10, 2, inputs=("y:0",))]
        with self.assertRaises(ValidationError):
            select_optimal(dup, 100, node_budget=10)
        # explicit dependency absent from the candidate set
        missing = [make_tx("a", 10, 1, inputs=("x:0",), depends=("ghost",))]
        with self.assertRaises(MissingDependencyError):
            select_optimal(missing, 100, node_budget=10)
        # invalid field inside a candidate
        bad = [make_tx("a", 10, 1, inputs=("x:0",)), "not-a-tx"]
        with self.assertRaises(ValidationError):
            select_optimal(bad, 100, node_budget=10)

    def test_candidate_limit(self):
        txs = [make_tx(f"t{i:02d}", 1, 1, inputs=(f"c{i}:0",))
               for i in range(MAX_CANDIDATES + 1)]
        with self.assertRaises(ValidationError) as ctx:
            select_optimal(txs, 10_000, node_budget=10)
        self.assertEqual(ctx.exception.kind, "capacity")
        # exactly at the limit is accepted
        r = select_optimal(txs[:MAX_CANDIDATES], 10_000, node_budget=10**9)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(len(r.selected), MAX_CANDIDATES)

    def test_ledger_is_consistent(self):
        pool = _build_pool([
            make_tx("p", 10, 1, inputs=("c:0",)),
            make_tx("h", 10, 9, inputs=("p:0",)),
        ])
        r = select_optimal(pool, 25, node_budget=1000)
        self.assertEqual([e.txid for e in r.ledger], list(r.selected))
        run_f = run_w = 0
        for e in r.ledger:
            run_f += e.fee
            run_w += e.weight
            self.assertEqual(e.cumulative_fee, run_f)
            self.assertEqual(e.cumulative_weight, run_w)
        self.assertEqual(run_f, r.total_fee)
        self.assertEqual(run_w, r.total_weight)


class TestOptimalSemantics(unittest.TestCase):
    def test_low_fee_ancestor_high_fee_child(self):
        # The high-fee child drags its low-fee ancestor into the package.
        pool = _build_pool([
            make_tx("p", 10, 1, inputs=("c:0",)),
            make_tx("k", 10, 8, inputs=("d:0",)),
            make_tx("h", 10, 9, inputs=("p:0",)),
        ])
        r = select_optimal(pool, 25, node_budget=10_000)
        self.assertEqual(list(r.selected), ["p", "h"])
        self.assertEqual(r.total_fee, 10)
        self.assertEqual(r.total_weight, 20)
        self.assertEqual(r.status, STATUS_OPTIMAL)

    def test_shared_ancestor_counted_once(self):
        # Two high-fee children share one low-fee ancestor; the ancestor
        # weight is paid once, making the combined package optimal.
        pool = _build_pool([
            make_tx("root", 10, 1, inputs=("c:0",)),
            make_tx("left", 10, 20, inputs=("root:0",)),
            make_tx("right", 10, 20, inputs=("root:1",)),
            make_tx("solo", 10, 15, inputs=("d:0",)),
        ])
        r = select_optimal(pool, 30, node_budget=10_000)
        # {root,left,right} fee 41 w30 beats {solo,left?no anc} etc.
        self.assertEqual(set(r.selected), {"root", "left", "right"})
        self.assertEqual(r.total_fee, 41)
        self.assertEqual(r.total_weight, 30)

    def test_double_spend_pair_mutually_exclusive(self):
        # x and y spend the same coin: never co-selected, even though
        # both fit under the weight cap together.
        txs = [
            make_tx("x", 10, 5, inputs=("coin:0",)),
            make_tx("y", 10, 7, inputs=("coin:0",)),
            make_tx("z", 10, 1, inputs=("other:0",)),
        ]
        r = select_optimal(txs, 30, node_budget=10_000)
        self.assertEqual(set(r.selected), {"y", "z"})
        self.assertEqual(r.total_fee, 8)
        # A child of a conflicting tx inherits the exclusion.
        txs2 = txs + [make_tx("c", 5, 1, inputs=("x:0",))]
        r2 = select_optimal(txs2, 100, node_budget=10_000)
        self.assertNotIn("x", r2.selected)
        self.assertNotIn("c", r2.selected)
        self.assertIn("y", r2.selected)

    def test_tie_break_smaller_weight_first(self):
        # Equal fee; the lighter package wins before any id comparison.
        pool = _build_pool([
            make_tx("a", 6, 10, inputs=("ia:0",)),
            make_tx("b", 4, 10, inputs=("ib:0",)),
        ])
        r = select_optimal(pool, 9, node_budget=1000)
        self.assertEqual(list(r.selected), ["b"])
        self.assertEqual(r.total_weight, 4)
        # ...whereas the legacy interface breaks the same tie by id.
        legacy = select_package(pool, 9, node_budget=1000)
        self.assertEqual(list(legacy.selected), ["a"])

    def test_tie_break_id_tuple_last(self):
        # Equal fee and equal weight: smallest ascending-id tuple wins.
        pool = _build_pool([
            make_tx("b", 5, 10, inputs=("ib:0",)),
            make_tx("a", 5, 10, inputs=("ia:0",)),
        ])
        r = select_optimal(pool, 5, node_budget=1000)
        self.assertEqual(list(r.selected), ["a"])

    def test_pool_is_read_only(self):
        pool = _build_pool(gen_txs(5, 10))
        before = (sorted(pool.ids()),
                  {t: sorted(pool.parents(t)) for t in pool.ids()},
                  {t: sorted(pool.children(t)) for t in pool.ids()})
        r = select_optimal(pool, 100, node_budget=100_000)
        after = (sorted(pool.ids()),
                 {t: sorted(pool.parents(t)) for t in pool.ids()},
                 {t: sorted(pool.children(t)) for t in pool.ids()})
        self.assertEqual(before, after)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        # Nothing was marked confirmed: every tx is still in the pool.
        self.assertEqual(len(pool), 10)


class TestOptimalBudget(unittest.TestCase):
    def test_budget_exhausted_status_and_bound(self):
        pool = _build_pool(gen_txs(3, 14))
        r = select_optimal(pool, 120, node_budget=3)
        self.assertEqual(r.status, STATUS_BUDGET_EXHAUSTED)
        self.assertNotEqual(r.status, STATUS_OPTIMAL)
        self.assertFalse(r.proven_optimal)
        self.assertLessEqual(r.nodes_used, 3)
        # The returned package is feasible ...
        chosen = set(r.selected)
        w = 0
        for tid in r.selected:
            self.assertTrue(pool.parents(tid) <= chosen)
            w += pool.get(tid).weight
        self.assertLessEqual(w, 120)
        # ... and the fee upper bound is admissible: it covers the
        # proven optimum computed with a generous budget.
        proven = select_optimal(pool, 120, node_budget=10**9)
        self.assertEqual(proven.status, STATUS_OPTIMAL)
        self.assertGreaterEqual(r.fee_upper_bound, proven.total_fee)
        self.assertGreaterEqual(r.fee_upper_bound, r.total_fee)
        self.assertLessEqual(r.total_fee, proven.total_fee)

    def test_generous_budget_proves_optimal(self):
        pool = _build_pool(gen_txs(3, 14))
        r = select_optimal(pool, 120, node_budget=10**9)
        self.assertEqual(r.status, STATUS_OPTIMAL)
        self.assertEqual(r.fee_upper_bound, r.total_fee)


class TestOptimalExhaustiveCrossCheck(unittest.TestCase):
    """select_optimal vs the independent <=16-tx exhaustive reference."""

    def test_seeded_random_dags_from_pool(self):
        cases = [
            (1, 8, 60), (2, 9, 80), (3, 10, 120), (4, 11, 100),
            (5, 12, 150), (9, 13, 90), (11, 14, 200), (13, 15, 120),
            (17, 16, 250),
        ]
        for seed, n, cap in cases:
            with self.subTest(seed=seed, n=n, cap=cap):
                txs = gen_txs(seed, n)
                pool = _build_pool(txs)
                r = select_optimal(pool, cap, node_budget=10**9)
                self.assertEqual(r.status, STATUS_OPTIMAL)
                fee, weight, ids = brute_force_conflict_optimal(txs, cap)
                self.assertEqual(r.total_fee, fee)
                self.assertEqual(r.total_weight, weight)
                self.assertEqual(set(r.selected), set(ids))

    def test_seeded_conflicting_candidate_lists(self):
        cases = [
            (1, 8, 60), (2, 10, 90), (3, 12, 120),
            (4, 14, 150), (5, 16, 200), (6, 15, 100),
        ]
        for seed, n, cap in cases:
            with self.subTest(seed=seed, n=n, cap=cap):
                txs = gen_txs_with_conflicts(seed, n)
                r = select_optimal(txs, cap, node_budget=10**9)
                self.assertEqual(r.status, STATUS_OPTIMAL)
                fee, weight, ids = brute_force_conflict_optimal(txs, cap)
                self.assertEqual(r.total_fee, fee)
                self.assertEqual(r.total_weight, weight)
                self.assertEqual(set(r.selected), set(ids))
                # Conflict-freedom double-checked directly.
                seen = set()
                for tid in r.selected:
                    tx = next(t for t in txs if t.txid == tid)
                    for op in tx.inputs:
                        self.assertNotIn(op, seen)
                        seen.add(op)

    def test_handcrafted_shared_ancestor_and_conflict(self):
        txs = [
            make_tx("root", 12, 2, inputs=("ext:0",)),
            make_tx("c1", 8, 30, inputs=("root:0",)),
            make_tx("c2", 8, 28, inputs=("root:1",)),
            make_tx("d1", 10, 25, inputs=("coin:0",)),
            make_tx("d2", 10, 26, inputs=("coin:0",)),  # conflicts with d1
            make_tx("solo", 20, 40, inputs=("free:0",)),
        ]
        for cap in (10, 20, 30, 38, 60, 1000):
            with self.subTest(cap=cap):
                r = select_optimal(txs, cap, node_budget=10**9)
                self.assertEqual(r.status, STATUS_OPTIMAL)
                fee, weight, ids = brute_force_conflict_optimal(txs, cap)
                self.assertEqual(r.total_fee, fee)
                self.assertEqual(r.total_weight, weight)
                self.assertEqual(set(r.selected), set(ids))


if __name__ == "__main__":
    unittest.main()
