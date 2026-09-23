import unittest

from ancestor_pool import (
    AncestorPool,
    make_tx,
    select_package,
    ValidationError,
    STATUS_PROVEN_OPTIMAL,
    STATUS_BUDGET_EXHAUSTED,
    CODE_OVERWEIGHT,
    CODE_ANCESTOR_EXCLUDED,
    CODE_NOT_SELECTED,
    CODE_BUDGET_INCONCLUSIVE,
)
from tests._reference import brute_force_optimal
from tests._data import gen_txs


def _build_pool(txs):
    pool = AncestorPool()
    pool.add_many(txs)
    return pool


class TestSelectBasic(unittest.TestCase):
    def test_empty_pool(self):
        r = select_package(AncestorPool(), 100, node_budget=100)
        self.assertEqual(r.selected, ())
        self.assertEqual(r.total_fee, 0)
        self.assertTrue(r.proven_optimal)

    def test_invalid_arguments(self):
        pool = _build_pool([make_tx("a", 10, 1, inputs=("x:0",))])
        with self.assertRaises(ValidationError):
            select_package(pool, 0, node_budget=10)
        with self.assertRaises(ValidationError):
            select_package(pool, True, node_budget=10)
        with self.assertRaises(ValidationError):
            select_package(pool, 100, node_budget=0)
        with self.assertRaises(ValidationError):
            select_package(pool, 100, node_budget=False)
        with self.assertRaises(ValidationError):
            select_package("not-a-pool", 100)

    def test_low_fee_parent_high_fee_child_package(self):
        # Taking the high-fee child forces the low-fee parent in.
        pool = _build_pool([
            make_tx("p", 10, 1, inputs=("c:0",)),
            make_tx("k", 10, 8, inputs=("d:0",)),  # independent rival
            make_tx("h", 10, 9, inputs=("p:0",)),  # high-fee child
        ])
        r = select_package(pool, 25, node_budget=10_000)
        # Closure choices: {p,h}=fee10 w20 ; {k}+? ; {p,h} best vs {k,p?}
        self.assertEqual(list(r.selected), ["p", "h"])
        self.assertEqual(r.total_fee, 10)
        self.assertTrue(r.proven_optimal)

    def test_overweight_transaction_exclusion(self):
        pool = _build_pool([
            make_tx("a", 100, 999, inputs=("x:0",)),
            make_tx("b", 10, 5, inputs=("y:0",)),
        ])
        r = select_package(pool, 50, node_budget=100)
        self.assertEqual(list(r.selected), ["b"])
        codes = {e.txid: e.code for e in r.excluded}
        self.assertEqual(codes["a"], CODE_OVERWEIGHT)
        self.assertNotIn("b", codes)

    def test_ancestor_excluded_reason(self):
        # Parent fits alone but buying it plus the child exceeds the
        # cap; the empty package wins the zero-fee tie, so the parent is
        # a plain non-selection while the child is blocked by ancestry.
        pool = _build_pool([
            make_tx("p", 40, 0, inputs=("x:0",)),
            make_tx("c", 20, 100, inputs=("p:0",)),
        ])
        r = select_package(pool, 50, node_budget=100)
        self.assertEqual(list(r.selected), [])
        codes = {e.txid: e.code for e in r.excluded}
        self.assertEqual(codes["p"], CODE_NOT_SELECTED)
        self.assertEqual(codes["c"], CODE_ANCESTOR_EXCLUDED)

    def test_output_is_topological(self):
        pool = _build_pool(gen_txs(7, 9))
        r = select_package(pool, 120, node_budget=50_000)
        pos = {tid: i for i, tid in enumerate(r.selected)}
        for tid in r.selected:
            for par in pool.parents(tid):
                if par in pos:
                    self.assertLess(pos[par], pos[tid])


class TestTieBreak(unittest.TestCase):
    def test_lex_smaller_id_tuple_wins(self):
        # Two mutually exclusive equal-fee single txs; "a" tuple wins.
        pool = _build_pool([
            make_tx("b", 10, 5, inputs=("ib:0",)),
            make_tx("a", 10, 5, inputs=("ia:0",)),
        ])
        r = select_package(pool, 10, node_budget=100)
        self.assertEqual(list(r.selected), ["a"])

    def test_prefix_tuple_wins(self):
        # Equal-fee independent txs ("a" fee 5, "b" fee 0): at cap 20
        # both {a} and {a,b} have fee 5; under plain tuple lex order the
        # proper prefix ("a",) is smaller, so the zero-fee extra tx is
        # not packed.
        pool = _build_pool([
            make_tx("a", 10, 5, inputs=("ia:0",)),
            make_tx("b", 10, 0, inputs=("ib:0",)),
        ])
        r = select_package(pool, 10, node_budget=100)
        self.assertEqual(list(r.selected), ["a"])
        r2 = select_package(pool, 20, node_budget=100)
        self.assertEqual(list(r2.selected), ["a"])
        # Positive fee on b breaks the fee tie in favour of taking both.
        pool2 = _build_pool([
            make_tx("a", 10, 5, inputs=("ia:0",)),
            make_tx("b", 10, 1, inputs=("ib:0",)),
        ])
        r3 = select_package(pool2, 20, node_budget=100)
        self.assertEqual(list(r3.selected), ["a", "b"])

    def test_tie_matches_tuple_semantics(self):
        # Independent txs with crafted fees to produce multi-way ties.
        pool = _build_pool([
            make_tx("a", 6, 5, inputs=("ea:0",)),
            make_tx("b", 6, 5, inputs=("eb:0",)),
            make_tx("c", 5, 5, inputs=("ec:0",)),
        ])
        r = select_package(pool, 11, node_budget=200)
        # cap 11: only a+c and b+c fit as pairs, both fee 10;
        # ("a","c") < ("b","c").
        self.assertEqual(list(r.selected), ["a", "c"])


class TestExhaustiveCrossCheck(unittest.TestCase):
    """Core solver vs the independent <=16-tx exhaustive reference."""

    def test_seeded_random_dags(self):
        cases = [
            (1, 8, 60), (2, 9, 80), (3, 10, 120), (4, 11, 100),
            (5, 12, 150), (9, 13, 90), (11, 14, 200), (13, 15, 120),
            (17, 16, 250),
        ]
        for seed, n, cap in cases:
            with self.subTest(seed=seed, n=n, cap=cap):
                txs = gen_txs(seed, n)
                pool = _build_pool(txs)
                r = select_package(pool, cap, node_budget=10_000_000)
                self.assertEqual(r.status, STATUS_PROVEN_OPTIMAL)
                fee, ids, weight = brute_force_optimal(txs, cap)
                self.assertEqual(r.total_fee, fee)
                # The reference returns the chosen set in id-sorted
                # order; the core returns it in topological order. They
                # must agree as sets.
                self.assertEqual(set(r.selected), set(ids))
                self.assertEqual(r.total_weight, weight)
                # Closure property on top of the reference agreement.
                chosen = set(r.selected)
                for tid in chosen:
                    self.assertTrue(pool.parents(tid) <= chosen)

    def test_handcrafted_low_parent_high_child(self):
        txs = [
            make_tx("p0", 30, 1, inputs=("ext:0",)),
            make_tx("c0", 5, 50, inputs=("p0:0",)),
            make_tx("solo", 20, 40, inputs=("coin:0",)),
            make_tx("p1", 10, 2, inputs=("coin2:0",)),
            make_tx("c1", 10, 20, inputs=("p1:0",)),
        ]
        pool = _build_pool(txs)
        for cap in (10, 25, 40, 55, 1000):
            with self.subTest(cap=cap):
                r = select_package(pool, cap, node_budget=1_000_000)
                fee, ids, _ = brute_force_optimal(txs, cap)
                self.assertEqual(r.total_fee, fee)
                # Reference returns the set id-sorted; core returns it
                # in topological order; compare as sets.
                self.assertEqual(set(r.selected), set(ids))


class TestBudget(unittest.TestCase):
    def test_tiny_budget_is_separate_status(self):
        pool = _build_pool(gen_txs(3, 14))
        r = select_package(pool, 120, node_budget=3)
        self.assertEqual(r.status, STATUS_BUDGET_EXHAUSTED)
        self.assertLessEqual(r.nodes_used, 3)
        # The returned package must still be feasible even if unproved.
        chosen = set(r.selected)
        w = 0
        for tid in r.selected:
            self.assertTrue(pool.parents(tid) <= chosen)
            w += pool.get(tid).weight
        self.assertLessEqual(w, 120)
        # Unsettled exclusions are labelled, not asserted as definite.
        codes = {e.txid: e.code for e in r.excluded}
        self.assertIn(CODE_BUDGET_INCONCLUSIVE, set(codes.values()))

    def test_budget_exhaustion_differs_from_proof(self):
        pool = _build_pool(gen_txs(3, 14))
        small = select_package(pool, 120, node_budget=5)
        full = select_package(pool, 120, node_budget=10_000_000)
        self.assertEqual(small.status, STATUS_BUDGET_EXHAUSTED)
        self.assertEqual(full.status, STATUS_PROVEN_OPTIMAL)
        # The bounded answer is a lower bound, never above the optimum.
        self.assertLessEqual(small.total_fee, full.total_fee)

    def test_proven_optimal_labels_exclusions_definitely(self):
        pool = _build_pool(gen_txs(5, 12))
        r = select_package(pool, 120, node_budget=10_000_000)
        self.assertTrue(r.proven_optimal)
        for e in r.excluded:
            self.assertNotEqual(e.code, CODE_BUDGET_INCONCLUSIVE)
            self.assertIn(
                e.code,
                (CODE_OVERWEIGHT, CODE_ANCESTOR_EXCLUDED, CODE_NOT_SELECTED),
            )


if __name__ == "__main__":
    unittest.main()
