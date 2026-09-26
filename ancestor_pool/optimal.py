"""Conflict-aware fee-optimal package selection for small candidate sets.

This module implements select_optimal, the *exact* small-pool selector,
alongside the older ancestor_pool.select_package fast interface. The
problem solved here is strictly richer:

Choose a subset S of at most 24 candidate transactions maximising total
fee subject to

* sum(weight) <= block_weight;
* S is ancestor-closed: selecting a transaction forces every
  not-yet-confirmed ancestor (explicit depends or spent creator
  outpoint) into S;
* S is conflict-free: two transactions spending the same outpoint
  (a double spend) can never coexist in S.

Tie-break (fixed, documented): among maximum-fee packages the one with
the **smaller total weight** wins; remaining ties are broken by the
lexicographically smallest tuple of selected ids sorted ascending
(tuple(sorted(ids_a)) < tuple(sorted(ids_b)); a proper prefix is
smaller). This differs deliberately from select_package, whose single
tie-break is the id tuple only.

The search is an exact branch-and-bound (never "fee-rate greedy then
patch ancestors"). A positive node_budget bounds the number of expanded
DFS nodes. The two terminal states are kept distinct:

* optimal -- the search completed; the package is the global optimum
  under the documented tie-break;
* budget_exhausted -- the budget ran out first. The best feasible
  package found so far is returned together with fee_upper_bound, an
  admissible integer upper bound on the optimum fee. The status is
  never reported as optimal in this case.

The pool is only read: selection never mutates the pool and never
marks any transaction as confirmed.
"""

from functools import cmp_to_key

from .errors import MissingDependencyError, ValidationError
from .graph import build_edges, closure_maps, topo_order
from .model import validate_tx
from .pool import AncestorPool
from .select import STATUS_BUDGET_EXHAUSTED, _check_int

MAX_CANDIDATES = 24
DEFAULT_BUDGET = 1_000_000

STATUS_OPTIMAL = "optimal"


class LedgerEntry:
    """One line of the fee/weight ledger for a selected transaction."""

    __slots__ = ("txid", "fee", "weight", "cumulative_fee", "cumulative_weight")

    def __init__(self, txid, fee, weight, cumulative_fee, cumulative_weight):
        self.txid = txid
        self.fee = fee
        self.weight = weight
        self.cumulative_fee = cumulative_fee
        self.cumulative_weight = cumulative_weight

    def __iter__(self):
        return iter(
            (
                self.txid,
                self.fee,
                self.weight,
                self.cumulative_fee,
                self.cumulative_weight,
            )
        )

    def __eq__(self, other):
        return isinstance(other, LedgerEntry) and tuple(self) == tuple(other)

    def __repr__(self):
        return (
            f"LedgerEntry(txid={self.txid!r}, fee={self.fee}, "
            f"weight={self.weight}, cumulative_fee={self.cumulative_fee}, "
            f"cumulative_weight={self.cumulative_weight})"
        )


class OptimalSelectionResult:
    """Result of select_optimal."""

    __slots__ = (
        "selected",
        "ledger",
        "total_fee",
        "total_weight",
        "fee_upper_bound",
        "status",
        "nodes_used",
        "node_budget",
        "block_weight",
    )

    def __init__(
        self,
        selected,
        ledger,
        total_fee,
        total_weight,
        fee_upper_bound,
        status,
        nodes_used,
        node_budget,
        block_weight,
    ):
        self.selected = tuple(selected)
        self.ledger = tuple(ledger)
        self.total_fee = total_fee
        self.total_weight = total_weight
        self.fee_upper_bound = fee_upper_bound
        self.status = status
        self.nodes_used = nodes_used
        self.node_budget = node_budget
        self.block_weight = block_weight

    @property
    def proven_optimal(self):
        return self.status == STATUS_OPTIMAL

    def __repr__(self):
        return (
            f"OptimalSelectionResult(status={self.status!r}, "
            f"selected={list(self.selected)!r}, total_fee={self.total_fee}, "
            f"total_weight={self.total_weight}, "
            f"fee_upper_bound={self.fee_upper_bound}, "
            f"nodes_used={self.nodes_used})"
        )


def _candidates_from_iterable(candidates):
    """Validate a standalone candidate list; return (txs, parents).

    Unlike ancestor_pool.validate_batch, intra-set double spends are
    *allowed* here: they are conflicts the selector must respect, not
    structural errors. Duplicate ids, unresolvable explicit
    dependencies, and dependency cycles are still rejected.
    """
    if isinstance(candidates, (str, bytes)) or not hasattr(
        candidates, "__iter__"
    ):
        raise ValidationError(
            f"candidates must be an AncestorPool or an iterable of Tx; "
            f"got {type(candidates).__name__}",
            "candidates",
            kind="type_error",
        )
    candidates = list(candidates)
    txs = {}
    for i, tx in enumerate(candidates):
        loc = f"candidates[{i}]"
        validate_tx(tx, loc)
        if tx.txid in txs:
            raise ValidationError(
                f"duplicate txid {tx.txid!r} in candidates",
                f"{loc}.txid",
                kind="duplicate",
            )
        txs[tx.txid] = tx
    for tx in txs.values():
        missing = sorted(d for d in tx.depends if d not in txs)
        if missing:
            raise MissingDependencyError(
                f"transaction {tx.txid!r} depends on ids absent from the "
                f"candidate set: {missing}",
                missing=missing,
                loc="candidates.depends",
            )
    parents, _children = build_edges(txs)
    # Raises ValidationError(kind="cycle") if the dependency graph cycles.
    topo_order(txs, parents)
    return txs, parents


def _conflict_sets(txs):
    """Return dict[txid -> set[txid]] of double-spend conflicts."""
    spenders = {}
    for tid, tx in txs.items():
        for op in tx.inputs:
            spenders.setdefault(op, set()).add(tid)
    conflicts = {tid: set() for tid in txs}
    for owners in spenders.values():
        if len(owners) > 1:
            for tid in owners:
                conflicts[tid].update(owners - {tid})
    return conflicts


def select_optimal(candidates, block_weight, *, node_budget=DEFAULT_BUDGET):
    """Compute the conflict-free ancestor-closed max-fee package.

    Args:
        candidates: an AncestorPool (read only) or an iterable of Tx.
            At most MAX_CANDIDATES (24) transactions. A plain iterable
            may contain mutually conflicting (double spending)
            transactions; a pool never does.
        block_weight: positive int weight capacity.
        node_budget: positive int; maximum number of DFS nodes expanded
            (one unit per expansion).

    Returns:
        OptimalSelectionResult. status is "optimal" only when the
        search completed; on "budget_exhausted" the returned package is
        the best feasible one found and fee_upper_bound is an
        admissible upper bound on the optimum fee.
    """
    if isinstance(candidates, AncestorPool):
        # Read-only snapshot; the pool is never mutated or marked.
        txs, parents = candidates.edge_maps()
    else:
        txs, parents = _candidates_from_iterable(candidates)
    _check_int(block_weight, "block_weight", "block_weight")
    if block_weight <= 0:
        raise ValidationError(
            f"block_weight must be positive; got {block_weight}",
            "block_weight",
        )
    _check_int(node_budget, "node_budget", "node_budget")
    if node_budget <= 0:
        raise ValidationError(
            f"node_budget must be positive; got {node_budget}",
            "node_budget",
        )

    n = len(txs)
    if n > MAX_CANDIDATES:
        raise ValidationError(
            f"candidate count {n} exceeds limit {MAX_CANDIDATES}",
            "candidates",
            kind="capacity",
        )
    if n == 0:
        return OptimalSelectionResult(
            (), (), 0, 0, 0, STATUS_OPTIMAL, 0, node_budget, block_weight
        )

    ancestors, _desc = closure_maps(txs, parents)
    conflicts = _conflict_sets(txs)
    order = topo_order(txs, parents)  # DFS decision order
    weight = [txs[tid].weight for tid in order]
    fee = [txs[tid].fee for tid in order]
    par = [
        frozenset(
            k for k, anc_id in enumerate(order) if anc_id in ancestors[tid]
        )
        for tid in order
    ]
    conf = [
        frozenset(
            k for k, other in enumerate(order) if other in conflicts[tid]
        )
        for tid in order
    ]
    ids_sorted = sorted(txs)

    # Descending fee/weight order, exact cross multiplication; ties by id.
    def ratio_cmp(a, b):
        lhs = fee[a] * weight[b]
        rhs = fee[b] * weight[a]
        if lhs != rhs:
            return -1 if lhs > rhs else 1
        return -1 if order[a] < order[b] else (
            1 if order[a] > order[b] else 0
        )

    ratio_order = sorted(range(n), key=cmp_to_key(ratio_cmp))

    counters = {"nodes": 0, "exhausted": False}
    state = [-1] * n  # -1 undecided, 0 excluded, 1 selected

    best_fee = 0
    best_weight = 0
    best_tuple = ()
    # Admissible upper bound on the optimum fee, maintained over the
    # unexplored frontier; exact (== best_fee) once the search completes.
    upper_bound = {"value": 0}

    def better(total_fee, total_weight, tup):
        if total_fee != best_fee:
            return total_fee > best_fee
        if total_weight != best_weight:
            return total_weight < best_weight
        return tup < best_tuple

    def consider(sel, total_fee, total_weight):
        nonlocal best_fee, best_weight, best_tuple
        tup = tuple(tid for tid in ids_sorted if tid in sel)
        if better(total_fee, total_weight, tup):
            best_fee = total_fee
            best_weight = total_weight
            best_tuple = tup

    def relaxation_bound(i, cap_left, cur_fee):
        """Integer floor of the fractional-knapsack bound on remaining fee.

        Closure and conflicts between undecided nodes are ignored, so
        the relaxed value is always >= every feasible completion's fee;
        flooring keeps the inequality admissible for integer optima.
        """
        bound = cur_fee
        cap = cap_left
        for j in ratio_order:
            if j < i or state[j] != -1:
                continue
            w = weight[j]
            if w <= cap:
                cap -= w
                bound += fee[j]
                if cap == 0:
                    break
            else:
                bound += (fee[j] * cap) // w
                break
        return bound

    def dfs(i, sel, cur_weight, cur_fee):
        if counters["exhausted"]:
            return
        bound = relaxation_bound(i, block_weight - cur_weight, cur_fee)
        if counters["nodes"] >= node_budget:
            # The next expansion would exceed the explicit budget. This
            # frontier node's bound covers its whole unexplored subtree.
            counters["exhausted"] = True
            if bound > upper_bound["value"]:
                upper_bound["value"] = bound
            return
        counters["nodes"] += 1

        if i == n:
            consider(sel, cur_fee, cur_weight)
            return

        # Strict prune only: an equal bound may still hide a lighter or
        # lexicographically smaller equal-fee package.
        if bound < best_fee:
            return

        node = i
        if any(state[p] == 0 for p in par[node]) or any(
            state[c] == 1 for c in conf[node]
        ):
            # An ancestor is out (closure) or a conflicting tx is in
            # (double spend): this node cannot be selected.
            state[node] = 0
            dfs(i + 1, sel, cur_weight, cur_fee)
            state[node] = -1
            if counters["exhausted"] and bound > upper_bound["value"]:
                upper_bound["value"] = bound
            return

        fits = cur_weight + weight[node] <= block_weight

        # Select branch first to build a strong incumbent quickly.
        if fits:
            state[node] = 1
            sel.add(order[node])
            dfs(
                i + 1,
                sel,
                cur_weight + weight[node],
                cur_fee + fee[node],
            )
            sel.discard(order[node])
            state[node] = -1
            if counters["exhausted"]:
                if bound > upper_bound["value"]:
                    upper_bound["value"] = bound
                return

        state[node] = 0
        dfs(i + 1, sel, cur_weight, cur_fee)
        state[node] = -1
        if counters["exhausted"] and bound > upper_bound["value"]:
            upper_bound["value"] = bound

    # Initial feasible incumbent: repeatedly take the highest-ratio
    # node whose parents are chosen, which conflicts with nothing
    # chosen, and that still fits.
    cap = block_weight
    g_fee = 0
    chosen = [False] * n
    remaining = set(range(n))
    greedy_sel = set()
    greedy_weight = 0
    while remaining:
        pick = None
        for j in ratio_order:
            if (
                j in remaining
                and all(chosen[p] for p in par[j])
                and not any(chosen[c] for c in conf[j])
                and weight[j] <= cap
            ):
                pick = j
                break
        if pick is None:
            break
        chosen[pick] = True
        remaining.discard(pick)
        greedy_sel.add(order[pick])
        cap -= weight[pick]
        greedy_weight += weight[pick]
        g_fee += fee[pick]
    consider(greedy_sel, g_fee, greedy_weight)

    dfs(0, set(), 0, 0)

    chosen_ids = set(best_tuple)
    total_fee = sum(txs[t].fee for t in chosen_ids)
    total_weight = sum(txs[t].weight for t in chosen_ids)
    # Lexicographically smallest topological order *of the chosen subset*.
    selected_topo = topo_order(txs, parents, subset=chosen_ids)

    ledger = []
    run_fee = 0
    run_weight = 0
    for tid in selected_topo:
        run_fee += txs[tid].fee
        run_weight += txs[tid].weight
        ledger.append(
            LedgerEntry(
                tid, txs[tid].fee, txs[tid].weight, run_fee, run_weight
            )
        )

    if counters["exhausted"]:
        status = STATUS_BUDGET_EXHAUSTED
        fee_upper = max(upper_bound["value"], total_fee)
    else:
        status = STATUS_OPTIMAL
        fee_upper = total_fee
    return OptimalSelectionResult(
        selected_topo,
        ledger,
        total_fee,
        total_weight,
        fee_upper,
        status,
        counters["nodes"],
        node_budget,
        block_weight,
    )
