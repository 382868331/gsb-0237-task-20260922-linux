# Ancestor-Package Constrained Mempool Replacement

A small, dependency-free, offline Python library implementing a
**task-specific subset** of ancestor-package constrained transaction-pool
replacement and fee-optimal block-package selection over a synthetic
transaction DAG.

- Python: **3.14.7** (standard library only; no third-party packages)
- OS: Linux native; no network, accounts, keys, database, middleware,
  cloud API, or Docker
- Interface bound: at most **30 transactions** in a pool; the
  independent exhaustive reference supports **≤ 16 transactions**

This implements only the finite mathematical/protocol subset defined by
this task. It is **not** a full node mempool and makes no claim of
Bitcoin/BIP125 or any other industry-standard wire compatibility.

## Layout

```
ancestor_pool/          importable library
  errors.py             error hierarchy + stable reason codes
  model.py              Tx / make_tx + field validation
  graph.py              DAG edges, closures, cycles, topological order,
                        batch and package structural validation
  pool.py               AncestorPool: atomic add / add_many / add_package
  select.py             select_package: budgeted branch-and-bound
tests/
  _reference.py         independent <=16-tx exhaustive reference
                        (does NOT import or call the core solver)
  _data.py              deterministic seeded synthetic DAG generator
  test_*.py             unit + cross-check tests
demo.py                 fixed-input live demonstration
```

## Running

```bash
python -m unittest discover -s tests -v   # tests
python demo.py                             # demonstration (< 1 s here)
```

No `pip install` step is required beyond a Python 3.14.7 interpreter.
`requirements.txt` is intentionally empty of dependencies.

## Public data structures

### `Tx`

```python
Tx(txid: str, weight: int, fee: int,
   inputs: tuple[str, ...] = (), depends: tuple[str, ...] = ())
```

Created via the validating factory `make_tx(...)`.

- `txid` — non-empty `str` without NUL; unique within a pool/batch.
- `weight` — positive `int`, ≤ 4,000,000.
- `fee` — non-negative `int`, ≤ 2⁶³−1.
- `inputs` — consumed outpoints as non-empty strings. Two transactions
  spending the identical outpoint string conflict.
- `depends` — explicit parent tx ids, in addition to parents inferred
  through outpoints.

Outpoint form: `"creator_id:vout"`. The creator is the substring before
the **last** `":"`; spending `creator_id:vout` creates a parent edge to
the tx with id `creator_id` when such a tx exists. An outpoint whose
prefix matches no known tx is an **external coin**: it has no edge and
no dependency requirement, but still participates in conflict
detection. Bare names without `":"` are always external coins.

All money/weight values are integers. Floats are rejected for `weight`
and `fee`, and `bool` is rejected for every integer parameter, so NaN
and Infinity cannot enter the system.

### Errors

All derive from `MempoolError`:

- `ValidationError(message, loc, kind)` — invalid field; `loc` pinpoints
  it (e.g. `"txs[3].fee"`); input is never silently corrected.
- `DuplicateTxIdError`, `MissingDependencyError(missing=...)`
- `ReplacementRejectedError(incoming_id, removed_ids, reason, detail)`
  with stable `reason` codes:
  - `fee_not_strictly_greater`
  - `fee_rate_not_strictly_higher`
  - `depends_on_removed_set`
  - `replacement_disallowed` (when `allow_replacement=False`)

## Replacement semantics — `AncestorPool.add(tx)`

If the incoming tx spends an outpoint already spent in the pool, let the
directly conflicting txs be *C* and the **removed set** be *C* plus every
descendant of *C* (transitively — these become orphans). The tx is
accepted only when **all** hold:

1. it does not depend (via `depends` or a spent creator outpoint) on any
   tx in the removed set;
2. `fee(incoming) > Σ fee(removed)` — strict;
3. `fee(incoming)/weight(incoming) > Σ fee(removed)/Σ weight(removed)`
   — strict, compared by exact integer cross multiplication
   `f_in·w_rm > f_rm·w_in`; no floating point.

`add` is **atomic**: a rejected or failing call leaves the pool
byte-for-byte unchanged (the evicted set, outpoint ownership, edges, and
closures are all restored). A non-conflicting tx is appended after its
dependencies are verified present. `add_many(txs)` validates the entire
batch against the union with the pool first, then applies it on a
scratch copy and adopts it only if every step succeeds — a late failure
leaves no partial change.

Read-only helpers: `len`, `in`, `ids()`, `get()`, `parents()`,
`children()`, `ancestors()`, `descendants()`, `spender_of(outpoint)`.
Capacity is 30 txs; a conflicting add at capacity is a replacement
(removes ≥ 1), not growth.

## Ancestor-package submission — `AncestorPool.add_package(txs, *, max_size=8)`

Submits **one dependency package of 1–8 transactions atomically**: the
given order is irrelevant (a parent may sit at any position, including
after its children — edges are resolved over pool ∪ package in one
pass), the whole package is validated as a DAG together, and it enters
the pool only when **every** check passes. On any failure nothing from
the package is kept: the live pool's txs, edges, closures and outpoint
ownership are byte-for-byte unchanged (all checks run before the
scratch copy is adopted).

Checks, each with a locatable `ValidationError`/`MissingDependencyError`:

- size must be within `1..max_size` (`kind="empty_package"` /
  `"package_too_large"`); `max_size` is keyword-only, an `int` in
  `1..8` (`MAX_PACKAGE_SIZE`);
- accepting the package must keep the pool ≤ 30 (`kind="capacity"`);
- per-field validity with indexed locations `txs[i].field`;
- duplicate ids inside the package or against the pool
  (`kind="duplicate"`);
- the same outpoint spent by two package txs (`kind="double_spend"`);
- spending an outpoint already spent **in the pool** — package
  submission is **append-only and never replaces**: this raises
  `kind="conflict_with_pool"`. Conflict replacement still goes through
  the single-tx `add` (with its fee/fee-rate rules);
- an explicit `depends` parent missing from both pool and package
  raises `MissingDependencyError(missing=(...))` with loc
  `txs[i].depends[j]`. Spending an outpoint whose creator prefix
  matches no known tx remains an external coin and needs no parent;
- any dependency cycle over pool ∪ package (`kind="cycle"`), including
  cycles formed purely through spent creator outpoints.

Returns `PackageResult`:

- `order` — the package txs in the lexicographically-smallest
  topological order (parents within the package first; pool parents
  need no position because they are already stored);
- `ids` (frozenset), `added_fee`, `added_weight`, `len(result)`.

`add` and `add_many` are unchanged and remain available; a later
package can build on an earlier one or on pool transactions. An
iterable (e.g. a generator) is accepted; a bare `str` is rejected.

## Package selection — `select_package(pool, block_weight, *, node_budget)`

Finds the dependency-closed (ancestor-closed) subset maximising total
fee subject to `Σ weight ≤ block_weight`.

- Search: explicit depth-first **branch-and-bound**. The bound is the
  fractional-knapsack relaxation over undecided nodes (closure ignored,
  so it stays an admissible upper bound); ratios are ordered by integer
  cross multiplication and the single fractional split item is floored
  with integer arithmetic, so an optimal branch can never be pruned.
  The problem is NP-hard, which is why the explicit budget exists.
- **Budget**: `node_budget` is a positive `int` that must always be
  passed (default `200_000`); exactly one DFS node expansion is charged
  per unit, and `nodes_used ≤ node_budget` always.
- **The two terminal states are deliberately distinct**:
  - `status="proven_optimal"` — search completed; the package is the
    global optimum;
  - `status="budget_exhausted"` — the budget ran out first; the best
    feasible package found is returned **without any optimality claim**.
  They are never conflated: budget exhaustion is not reported as
  "no solution".
- Tie-break: among equal-fee packages, the lexicographically smallest
  tuple of selected ids written in ascending order wins, using standard
  Python tuple comparison (`tuple(sorted(ids_a)) < tuple(sorted(ids_b))`;
  a proper prefix is smaller). Consequence: at strictly equal fee a
  zero-fee extra transaction is not packed when its inclusion only
  lengthens the tuple.
- `SelectionResult.selected` lists included txs in the
  **lexicographically-smallest topological order** of the chosen subset
  (heap-based Kahn over the induced subgraph).
- Every excluded tx carries one stable reason code:
  - `overweight_even_alone` — its weight exceeds `block_weight`;
  - `ancestor_excluded` — one or more ancestors are out;
  - `not_selected` — it fits with all ancestors but is not part of the
    optimal tie-broken package;
  - `budget_inconclusive` — only under `budget_exhausted`: its status
    could not be settled within budget (never used when proven).

## Numeric accuracy / error policy

There are **no floating-point results** in the core: fees, weights,
fee comparisons, fee-rate comparisons, totals, and the search bound are
all exact integer/rational comparisons, so the accuracy tolerance is
**zero** for every reported number and comparison. No tolerance exists
that could hide a boundary error; equal fee and equal fee-rate are
rejected (strict inequalities), as exercised by the tests. The only
division anywhere is in test/demo diagnostics that merely print
human-readable rates.

## Design trade-offs

- Integer-only money model and exact cross multiplication deliberately
  chosen over float rates to make strict-boundary behaviour exact.
- Branch-and-bound with an incumbent seeded by a feasible ratio-greedy
  package; strong pruning keeps the ≤30-tx interface fast in practice
  while the node budget guarantees a hard, caller-visible work bound.
- Removed set includes **all** descendants of any direct conflict
  (orphan eviction), even when a descendant is individually high fee;
  this matches the stated rule and is covered by a test.
- External coins are syntactic (`"creator:vout"` with no matching
  creator) rather than modelled as UTXOs, sufficient for the synthetic
  DAG contract.

## Known limitations

- Fixed 30-tx pool cap; beyond the bound the solver offers no
  optimality guarantee except what the node budget permits.
- A tiny `node_budget` can return `budget_exhausted` immediately; the
  returned package is always feasible but may be suboptimal and its
  tie-break unsettled (reported honestly via status and exclusion
  codes).
- No signatures, scripts, consensus rules, timelocks, mempool policy
  knobs (signaling, BIP125 rules, package limits beyond count), or
  persistence; in-memory single-process only.
- Tie-break is exactly one documented convention (ascending-id tuple,
  prefix-smallest); other lexicographic conventions are not offered.
- Package submission is append-only: a package that conflicts with an
  already-pooled outpoint is rejected rather than applying RBF as a
  batch; package-level replacement with aggregate fee/fee-rate policy,
  chain-reorg re-feed, and memory eviction are intentionally out of
  scope for this iteration (single-tx replacement remains available).

## Demo

`python demo.py` computes live (no sleeps, no pre-recorded data): a
successful conflict replacement that evicts a low-fee parent and its
high-fee child, the proven fee-optimal package with three different
per-exclusion reasons, an actually raised
`ReplacementRejectedError(fee_rate_not_strictly_higher)` boundary with
the pool verified unchanged, a real `budget_exhausted` result, and the
atomic ancestor package: an unordered 8-tx DAG (the inclusive size
boundary) accepted in topological order, plus actually triggered
cycle / intra-package double-spend / pool-owned-outpoint /
missing-parent errors each verified to leave the pool unchanged.
