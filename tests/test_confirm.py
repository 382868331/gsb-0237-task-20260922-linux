"""Tests for ``AncestorPool.confirm`` (ledger-confirmed blocks).

The enumeration helper here is independent: it rebuilds parent edges
from raw ``Tx`` fields and enumerates every closed subset of the
surviving pool without calling the core solver, then checks
``select_package`` after confirmation.
"""

import unittest

from ancestor_pool import (
    AncestorPool,
    make_tx,
    select_package,
    ValidationError,
    ConfirmRejectedError,
    CONFIRM_DUPLICATE_ID,
    CONFIRM_ALREADY_CONFIRMED,
    CONFIRM_DOUBLE_SPEND,
    CONFIRM_SPENDS_CONFIRMED,
    CONFIRM_BAD_ORDER,
)


def _raw_parents(tx, by_id):
    ps = set(tx.depends)
    for op in tx.inputs:
        i = op.rfind(":")
        creator = op[:i] if i > 0 else None
        if creator is not None and creator in by_id and creator != tx.txid:
            ps.add(creator)
    return ps


def enumerate_best(txs, block_weight):
    """Independently enumerate the best closed pool-only subset."""
    by_id = {t.txid: t for t in txs}
    parents = {t.txid: _raw_parents(t, by_id) & set(by_id) for t in txs}
    ids = sorted(by_id)
    best = (0, (), 0)
    for bits in range(1 << len(ids)):
        chosen = {ids[k] for k in range(len(ids)) if (bits >> k) & 1}
        if any(not (parents[t] <= chosen) for t in chosen):
            continue
        w = sum(by_id[t].weight for t in chosen)
        if w > block_weight:
            continue
        f = sum(by_id[t].fee for t in chosen)
        tup = tuple(sorted(chosen))
        if f > best[0] or (f == best[0] and (not best[1] or tup < best[1])):
            best = (f, tup, w)
    return best


class TestConfirm(unittest.TestCase):
    def _parent_child_pool(self):
        p = AncestorPool()
        p.add(make_tx("p", 10, 5, inputs=("coin:0",)))
        p.add(make_tx("c", 20, 9, inputs=("p:0",)))
        return p

    def test_confirmed_parent_child_stays(self):
        pool = self._parent_child_pool()
        res = pool.confirm(
            [make_tx("p", 10, 5, inputs=("coin:0",))]
        )
        self.assertEqual(res.confirmed_ids, ("p",))
        self.assertEqual(res.evicted_ids, frozenset())
        self.assertEqual(res.retained_ids, frozenset({"c"}))
        self.assertEqual(res.reparented_ids, frozenset({"c"}))
        self.assertNotIn("p", pool)
        self.assertIn("c", pool)
        self.assertEqual(pool.parents("c"), frozenset())
        self.assertEqual(pool.confirmed_parents("c"), frozenset({"p"}))
        self.assertTrue(pool.is_outpoint_spent_confirmed("coin:0"))
        self.assertEqual(pool.confirmed_outpoints(), frozenset({"p:0"}))
        self.assertIn("p", pool.confirmed_ids())

    def test_select_ignores_confirmed_weight_and_fee(self):
        pool = self._parent_child_pool()
        pool.confirm([make_tx("p", 100, 1000, inputs=("coin:0",))])
        sel = select_package(pool, 15, node_budget=10_000)
        # c alone (weight 20) does not fit 15: closure is within the
        # pool, so p must not be packed; with budget 25 only c packs.
        self.assertEqual(set(sel.selected), set())
        sel25 = select_package(pool, 25, node_budget=10_000)
        self.assertEqual(list(sel25.selected), ["c"])
        self.assertEqual(sel25.total_weight, 20)
        self.assertEqual(sel25.total_fee, 9)

    def test_select_matches_independent_enumeration_after_confirm(self):
        # p confirmed; pool holds c (spends p:0), d (spends c:0) and an
        # independent k spending another coin.
        pool = AncestorPool()
        pool.add(make_tx("p", 40, 4, inputs=("coin:0",)))
        pool.add(make_tx("c", 10, 30, inputs=("p:0",)))
        pool.add(make_tx("d", 12, 7, inputs=("c:0",)))
        pool.add(make_tx("k", 9, 11, inputs=("other:0",)))
        pool.confirm([make_tx("p", 40, 4, inputs=("coin:0",))])
        remaining = [pool.get(t) for t in sorted(pool.ids())]
        for bw in (5, 9, 10, 19, 22, 30, 100):
            f, tup, w = enumerate_best(remaining, bw)
            sel = select_package(pool, bw, node_budget=100_000)
            self.assertEqual(sel.total_fee, f, bw)
            self.assertEqual(tuple(sorted(sel.selected)), tup, bw)
            self.assertEqual(sel.total_weight, w, bw)

    def test_out_of_block_conflict_evicts_descendants(self):
        # Pool: p spends coin:0; c spends p:0; g spends c:0.
        # Block tx b spends p:0 -> conflicts with c; c and g are evicted.
        pool = AncestorPool()
        pool.add(make_tx("p", 10, 5, inputs=("coin:0",)))
        pool.add(make_tx("c", 10, 5, inputs=("p:0",)))
        pool.add(make_tx("g", 10, 5, inputs=("c:0",)))
        res = pool.confirm([
            make_tx("p", 10, 5, inputs=("coin:0",)),
            make_tx("b", 10, 6, inputs=("p:0",)),
        ])
        self.assertEqual(set(res.evicted_ids), {"c", "g"})
        self.assertEqual(set(pool.ids()), set())
        self.assertEqual(set(res.confirmed_ids), {"p", "b"})
        self.assertTrue(pool.is_outpoint_spent_confirmed("p:0"))

    def test_conflict_without_same_id_does_not_touch_unrelated(self):
        pool = AncestorPool()
        pool.add(make_tx("p", 10, 5, inputs=("coin:0",)))
        pool.add(make_tx("c", 10, 5, inputs=("p:0",)))
        pool.add(make_tx("k", 10, 5, inputs=("other:0",)))
        # Block does NOT include p, but spends p:0: p is an ancestor of
        # c (which conflicts), so p is NOT in c's descendants and must
        # survive as a pool tx even though its output is now spent.
        res = pool.confirm([make_tx("b", 10, 9, inputs=("p:0",))])
        self.assertEqual(set(res.evicted_ids), {"c"})
        self.assertEqual(set(pool.ids()), {"p", "k"})

    def test_parent_and_child_confirmed_in_one_block(self):
        pool = self._parent_child_pool()
        res = pool.confirm([
            make_tx("p", 10, 5, inputs=("coin:0",)),
            make_tx("c", 20, 9, inputs=("p:0",)),
        ])
        self.assertEqual(res.confirmed_ids, ("p", "c"))
        self.assertEqual(res.evicted_ids, frozenset())
        self.assertEqual(res.retained_ids, frozenset())
        self.assertEqual(set(pool.ids()), set())
        self.assertEqual(set(pool.confirmed_ids()), {"p", "c"})
        self.assertEqual(pool.confirmed_outpoints(), frozenset({"p:0"}))

    def test_grandchild_survives_parent_child_confirmation(self):
        pool = AncestorPool()
        pool.add(make_tx("p", 10, 5, inputs=("coin:0",)))
        pool.add(make_tx("c", 10, 5, inputs=("p:0",)))
        pool.add(make_tx("g", 10, 5, inputs=("c:0",)))
        res = pool.confirm([
            make_tx("p", 10, 5, inputs=("coin:0",)),
            make_tx("c", 10, 5, inputs=("p:0",)),
        ])
        self.assertEqual(set(pool.ids()), {"g"})
        self.assertEqual(pool.parents("g"), frozenset())
        self.assertEqual(pool.confirmed_parents("g"), frozenset({"c"}))
        self.assertEqual(res.reparented_ids, frozenset({"g"}))
        # g now depends on a confirmed output and can still be selected.
        sel = select_package(pool, 100, node_budget=1000)
        self.assertEqual(list(sel.selected), ["g"])

    def test_invalid_blocks_roll_back_entirely(self):
        cases = [
            # duplicate id inside the block
            lambda: [
                make_tx("p", 10, 5, inputs=("coin:0",)),
                make_tx("p", 10, 6, inputs=("coin:1",)),
            ],
            # intra-block double spend
            lambda: [
                make_tx("a", 10, 5, inputs=("coin:0",)),
                make_tx("b", 10, 6, inputs=("coin:0",)),
            ],
            # bad dependency order (child listed before parent)
            lambda: [
                make_tx("c", 20, 9, inputs=("p:0",)),
                make_tx("p", 10, 5, inputs=("coin:0",)),
            ],
        ]
        reasons = [
            CONFIRM_DUPLICATE_ID,
            CONFIRM_DOUBLE_SPEND,
            CONFIRM_BAD_ORDER,
        ]
        for build, reason in zip(cases, reasons):
            pool = self._parent_child_pool()
            ids_before = pool.ids()
            owners_before = dict(pool.edge_maps()[0])
            with self.assertRaises(ConfirmRejectedError) as cm:
                pool.confirm(build())
            self.assertEqual(cm.exception.reason, reason)
            self.assertEqual(pool.ids(), ids_before)
            self.assertEqual(set(pool.edge_maps()[0]), set(owners_before))
            self.assertEqual(pool.confirmed_ids(), frozenset())

    def test_already_confirmed_txid_rejected_atomically(self):
        pool = self._parent_child_pool()
        pool.confirm([make_tx("p", 10, 5, inputs=("coin:0",))])
        ids_before = pool.ids()
        with self.assertRaises(ConfirmRejectedError) as cm:
            pool.confirm([make_tx("p", 10, 5, inputs=("coin:1",))])
        self.assertEqual(cm.exception.reason, CONFIRM_ALREADY_CONFIRMED)
        self.assertEqual(pool.ids(), ids_before)

    def test_block_spending_previously_confirmed_outpoint_rejected(self):
        pool = self._parent_child_pool()
        pool.confirm([make_tx("p", 10, 5, inputs=("coin:0",))])
        with self.assertRaises(ConfirmRejectedError) as cm:
            pool.confirm([make_tx("b", 10, 9, inputs=("coin:0",))])
        self.assertEqual(cm.exception.reason, CONFIRM_SPENDS_CONFIRMED)

    def test_pool_add_spending_confirmed_outpoint_rejected(self):
        pool = self._parent_child_pool()
        pool.confirm([make_tx("p", 10, 5, inputs=("coin:0",))])
        ids_before = pool.ids()
        with self.assertRaises(ValidationError):
            pool.add(make_tx("r", 10, 99, inputs=("coin:0",)))
        self.assertEqual(pool.ids(), ids_before)

    def test_field_errors_roll_back(self):
        pool = self._parent_child_pool()
        with self.assertRaises(ValidationError):
            pool.confirm(["not a tx"])
        self.assertEqual(set(pool.ids()), {"p", "c"})

    def test_non_iterable_rejected(self):
        pool = AncestorPool()
        with self.assertRaises(ValidationError):
            pool.confirm(42)

    def test_confirmed_tx_not_reported_as_evicted(self):
        pool = self._parent_child_pool()
        res = pool.confirm([
            make_tx("p", 10, 5, inputs=("coin:0",)),
            make_tx("c", 20, 9, inputs=("p:0",)),
        ])
        self.assertTrue(set(res.confirmed_ids).isdisjoint(res.evicted_ids))

    def test_two_blocks_accumulate_confirmed_state(self):
        pool = AncestorPool()
        pool.add(make_tx("a", 10, 5, inputs=("coinA:0",)))
        pool.add(make_tx("b", 10, 5, inputs=("coinB:0",)))
        pool.confirm([make_tx("a", 10, 5, inputs=("coinA:0",))])
        pool.confirm([make_tx("b", 10, 5, inputs=("coinB:0",))])
        self.assertEqual(set(pool.confirmed_ids()), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
