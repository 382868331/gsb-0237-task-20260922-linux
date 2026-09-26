"""Truly optimal ancestor-package selection with conflict exclusion.

This module adds :func:`select_optimal`, the incremental companion to
the older fast selector :func:`ancestor_pool.select.select_package`.
It solves a strictly richer problem over an explicit candidate set:

* candidates are **at most 24** transactions (interface bound);
* a selected transaction must include all of its **not-yet-confirmed
  ancestors** present in the candidate set (ancestor closure);
* **conflicting transactions cannot coexist**: two candidates spending
  the same outpoint string are mutually exclusive, even indirectly;
* the objective is maximum total fee under the weight cap; ties are
  broken by **smaller total weight first**, then by the **canonical
  transaction-id set** (the tuple of selected ids sorted ascending,
  compared with standard Python tuple ordering).

This is deliberately *not* "fee-rate greedy then backfill ancestors":
the solver is an exact branch-and-bound over the candidate DAG, so the
returned package is the true optimum whenever ``status == "optimal"``.

Budget and honesty
------------------
``node_budget`` bounds the number of expanded DFS nodes (one unit per
expansion). When the budget runs out, the result carries
``status="budget_exhausted"`` together with the best feasible package
found so far and ``fee_upper_bound`` -- an admissible upper bound on
the optimum fee (integer floor of the fractional-knapsack relaxation
over all candidates, closure and conflicts ignored). A budget-exhausted
result is never reported as ``"optimal"``.

Read-only guarantee
-------------------
The selection never mutates the pool: candidates are copied out via
:meth:`AncestorPool.edge_maps` (or validated from a plain iterable) and
no transaction is marked confirmed or otherwise annotated. The pool
object is byte-for-byte identical before and after the call.
"""

from functools import cmp_to_key

from .errors import MissingDependencyError, ValidationError
from .graph import build_edges, closure_maps, topo_order
from .model import validate_tx
from .pool import AncestorPool

MAX_CANDIDATES = 24
DEFAULT_NODE_BUDGET = 1_000_000

STATUS_OPTIMAL = "optimal"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"


class OptimalResult:
    """Result of :func:`select_optimal`.

    Attributes:
        selected: chosen txids in lexicographically-smallest
            topological order of the chosen subset (parents first).
        total_fee / total_weight: the fee/weight ledger of the package.
        status: ``"optimal"`` or ``"budget_exhausted"``.
        fee_upper_bound: admissible upper bound on the optimum fee;
            equals ``total_fee`` when ``status == "optimal"``.
        nodes_used / node_budget: search accounting.
        block_weight: the weight cap that was applied.
        candidates: sorted tuple of all candidate txids considered.
    """

    __slots__ = (
        "selected",
        "total_fee",
        "total_weight",
        "status",
        "fee_upper_bound",
        "nodes_used",
        "node_budget",
        "block_weight",
        "candidates",
    )

    def __init__(
        self,
        selected,
        total_fee,
        total_weight,
        status,
        fee_upper_bound,
        nodes_used,
        node_budget,
        block_weight,
        candidates,
    ):
        self.selected = tuple(selected)
        self.total_fee = total_fee
        self.total_weight = total_weight
        self.status = status
        self.fee_upper_bound = fee_upper_bound
        self.nodes_used = nodes_used
        self.node_budget = node_budget
        self.block_weight = block_weight
        self.candidates = tuple(candidates)

    @property
    def proven_optimal(self):
        return self.status == STATUS_OPTIMAL

    def __repr__(self):
        return (
            f"OptimalResult(status={self.status!r}, "
            f"selected={list(self.selected)!r}, "
            f"total_fee={self.total_fee}, "
            f"total_weight={self.total_weight}, "
            f"fee_upper_bound={self.fee_upper_bound}, "
            f"nodes_used={self.nodes_used})"
        )


def _check_int(value, name, loc):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an int (bool rejected); got "
            f"{type(value).__name__}",
            loc,
            kind="type_error",
        )


def _collect_candidates(source):
    """Return ``(txs, parents)`` maps without mutating ``source``.

    ``source`` is either an :class:`AncestorPool` (read via its
    read-only ``edge_maps`` copy) or a finite iterable of
    :class:`~ancestor_pool.model.Tx`. Unlike a pool batch, a plain
    candidate iterable **may** contain mutually conflicting
    (double-spending) transactions -- excluding them from coexisting is
    exactly the solver's job -- so only structural validation is done:
    field validity, id uniqueness, resolvable explicit dependencies,
    and acyclicity.
    """
    if isinstance(source, AncestorPool):
        txs, parents = source.edge_maps()
        return txs, parents
    if isinstance(source, (str, bytes)) or not hasattr(source, "__iter__"):
        raise ValidationError(
            f"candidates must be an AncestorPool or an iterable of Tx; "
            f"got {type(source).__name__}",
            "candidates",
            kind="type_error",
        )
    items = list(source)
    txs = {}
    for i, tx in enumerate(items):
        validate_tx(tx, f"candidates[{i}]")
        if tx.txid in txs:
            raise ValidationError(
                f"duplicate txid {tx.txid!r} in candidates",
                f"candidates[{i}]",
                kind="duplicate",
            )
        txs[tx.txid] = tx
    for i, tx in enumerate(items):
        missing = sorted(d for d in tx.depends if d not in txs)
        if missing:
            raise MissingDependencyError(
                f"candidate {tx.txid!r} depends on ids absent from the "
                f"candidate set: {missing}",
                missing=missing,
                loc=f"candidates[{i}].depends",
            )
    parents, _children = build_edges(txs)
    # Topological sort doubles as the dependency-cycle check.
    topo_order(txs, parents)
    return txs, parents


def select_optimal(source, block_weight, *, node_budget=DEFAULT_NODE_BUDGET):
    """Select the truly optimal ancestor-closed, conflict-free package.

    Args:
        source: an :class:`AncestorPool` (used read-only) or an
            iterable of :class:`Tx` candidates (at most 24). Plain
            iterables may contain double-spending candidates.
        block_weight: positive ``int`` weight capacity.
        node_budget: positive ``int``; maximum number of DFS node
            expansions. On exhaustion the best feasible package found
            is returned with ``status="budget_exhausted"`` and an
            admissible ``fee_upper_bound`` -- never as ``"optimal"``.

    Returns:
        OptimalResult.
    """
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

    txs, parents = _collect_candidates(source)
    n = len(txs)
    if n > MAX_CANDIDATES:
        raise ValidationError(
            f"at most {MAX_CANDIDATES} candidates supported; got {n}",
            "candidates",
            kind="capacity",
        )
    if n == 0:
        return OptimalResult(
            (), 0, 0, STATUS_OPTIMAL, 0, 0, node_budget, block_weight, ()
        )

    ancestors, _desc = closure_maps(txs, parents)
    order = topo_order(txs, parents)  # parents precede children
    pos = {tid: i for i, tid in enumerate(order)}
    weight = [txs[tid].weight for tid in order]
    fee = [txs[tid].fee for tid in order]
    par_idx = [frozenset(pos[a] for a in ancestors[tid]) for tid in order]
    # Outpoints each candidate spends (conflict exclusion is by exact
    # outpoint string, including external coins).
    spends = [txs[tid].inputs for tid in order]
    ids_sorted = sorted(txs)

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

    def consider(sel, cur_fee, cur_weight):
        nonlocal best_fee, best_weight, best_tuple
        tup = tuple(tid for tid in ids_sorted if tid in sel)
        if (
            cur_fee > best_fee
            or (
                cur_fee == best_fee
                and (
                    cur_weight < best_weight
                    or (cur_weight == best_weight and tup < best_tuple)
                )
            )
        ):
            best_fee = cur_fee
            best_weight = cur_weight
            best_tuple = tup

    def relaxation_bound(i, cap_left, cur_fee):
        """Admissible fee bound: fractional knapsack over undecided
        nodes, closure and conflicts ignored, integer-floored."""
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

    # Global admissible upper bound, reported when the budget runs out.
    root_bound = relaxation_bound(0, block_weight, 0)

    def dfs(i, sel, spent, cur_weight, cur_fee):
        if counters["exhausted"]:
            return
        if counters["nodes"] >= node_budget:
            counters["exhausted"] = True
            return
        counters["nodes"] += 1

        # Every prefix is feasible by construction (parents selected
        # first, conflicts rejected, weight cap enforced), so the
        # current selection is always a valid incumbent candidate.
        consider(sel, cur_fee, cur_weight)
        if i == n:
            return

        # Strict prune: a bound below the incumbent fee cannot win; a
        # bound equal to it can only tie on fee, and any extension has
        # strictly larger weight, so it cannot improve the tie-break
        # once the current weight already reaches the incumbent's.
        bound = relaxation_bound(i, block_weight - cur_weight, cur_fee)
        if bound < best_fee or (
            bound == best_fee and cur_weight >= best_weight
        ):
            return

        node = i
        blocked = any(state[p] == 0 for p in par_idx[node])
        if not blocked:
            fits = cur_weight + weight[node] <= block_weight
            clean = not any(op in spent for op in spends[node])
            if fits and clean:
                state[node] = 1
                sel.add(order[node])
                added = [op for op in spends[node] if op not in spent]
                for op in added:
                    spent.add(op)
                dfs(
                    i + 1,
                    sel,
                    spent,
                    cur_weight + weight[node],
                    cur_fee + fee[node],
                )
                for op in added:
                    spent.discard(op)
                sel.discard(order[node])
                state[node] = -1
                if counters["exhausted"]:
                    return

        state[node] = 0
        dfs(i + 1, sel, spent, cur_weight, cur_fee)
        state[node] = -1

    dfs(0, set(), set(), 0, 0)

    chosen_ids = set(best_tuple)
    total_fee = sum(txs[t].fee for t in chosen_ids)
    total_weight = sum(txs[t].weight for t in chosen_ids)
    selected_topo = topo_order(txs, parents, subset=chosen_ids)
    status = (
        STATUS_BUDGET_EXHAUSTED
        if counters["exhausted"]
        else STATUS_OPTIMAL
    )
    upper = (
        total_fee
        if status == STATUS_OPTIMAL
        else max(root_bound, total_fee)
    )
    return OptimalResult(
        selected_topo,
        total_fee,
        total_weight,
        status,
        upper,
        counters["nodes"],
        node_budget,
        block_weight,
        ids_sorted,
    )
