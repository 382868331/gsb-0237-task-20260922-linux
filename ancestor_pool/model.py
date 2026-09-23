"""Transaction data model and input validation.

Contract (fixed for this task)
------------------------------
* ``txid``      : non-empty ``str`` without NUL; unique within a batch/pool.
* ``weight``    : positive ``int``; ``bool`` is rejected. Upper bound
                  ``WEIGHT_LIMIT``.
* ``fee``       : non-negative ``int``; ``bool`` is rejected. Floats are
                  never accepted, so NaN/Infinity cannot enter and no
                  floating-point comparison is ever made by the core.
* ``inputs``    : ``tuple[str, ...]`` of consumed outpoints. Each is a
                  non-empty ``str`` without NUL. Outpoints are unique
                  within one transaction.
* ``depends``   : ``tuple[str, ...]`` of parent tx ids (in addition to
                  parent links inferred through shared outpoints). Ids
                  unique within the tuple, must not include ``txid``.

Fee rates (``fee / weight``) are compared by exact cross multiplication:
``f1*w2 > f2*w1``. There is therefore no rounding and no tolerance on
fees, weights, or fee-rate comparisons.
"""

from .errors import ValidationError

WEIGHT_LIMIT = 4_000_000
FEE_LIMIT = 2**63 - 1
MAX_INPUTS = 10_000


def _reject_bool_and_int(value, name, loc):
    # bool is a subclass of int; the contract rejects it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an int (bool rejected, float rejected); got "
            f"{type(value).__name__}",
            loc,
            kind="type_error",
        )


def _validate_str(value, name, loc, kind="type_error"):
    if not isinstance(value, str):
        raise ValidationError(
            f"{name} must be a str; got {type(value).__name__}", loc, kind=kind
        )
    if value == "":
        raise ValidationError(f"{name} must be non-empty", loc, kind=kind)
    if "\x00" in value:
        raise ValidationError(f"{name} must not contain NUL", loc, kind=kind)


def _validate_str_tuple(value, name, loc):
    if not isinstance(value, (tuple, list)):
        raise ValidationError(
            f"{name} must be a tuple or list of str; got "
            f"{type(value).__name__}",
            loc,
            kind="type_error",
        )
    seen = set()
    for i, item in enumerate(value):
        iloc = f"{loc}[{i}]"
        _validate_str(item, f"{name}[{i}]", iloc)
        if item in seen:
            raise ValidationError(
                f"{name} contains duplicate value {item!r}",
                iloc,
                kind="duplicate",
            )
        seen.add(item)
    return tuple(value)


class Tx:
    """Immutable transaction record.

    Construct via :func:`make_tx` (validating) or construct directly only
    with already-validated values. Instances are compared/hashed by
    identity-independent value, so they can be used as dict keys.
    """

    __slots__ = ("txid", "weight", "fee", "inputs", "depends")

    def __init__(self, txid, weight, fee, inputs=(), depends=()):
        self.txid = txid
        self.weight = weight
        self.fee = fee
        self.inputs = tuple(inputs)
        self.depends = tuple(depends)

    def __eq__(self, other):
        if not isinstance(other, Tx):
            return NotImplemented
        return (
            self.txid == other.txid
            and self.weight == other.weight
            and self.fee == other.fee
            and self.inputs == other.inputs
            and self.depends == other.depends
        )

    def __hash__(self):
        return hash(
            (self.txid, self.weight, self.fee, self.inputs, self.depends)
        )

    def __repr__(self):
        return (
            f"Tx(txid={self.txid!r}, weight={self.weight}, fee={self.fee}, "
            f"inputs={self.inputs!r}, depends={self.depends!r})"
        )


def validate_tx(tx, loc="tx"):
    """Validate every public field of ``tx`` in place (returns ``None``)."""
    if not isinstance(tx, Tx):
        raise ValidationError(
            f"expected Tx; got {type(tx).__name__}", loc, kind="type_error"
        )
    _validate_str(tx.txid, "txid", f"{loc}.txid")
    _reject_bool_and_int(tx.weight, "weight", f"{loc}.weight")
    if tx.weight <= 0:
        raise ValidationError(
            f"weight must be a positive int; got {tx.weight}",
            f"{loc}.weight",
        )
    if tx.weight > WEIGHT_LIMIT:
        raise ValidationError(
            f"weight {tx.weight} exceeds limit {WEIGHT_LIMIT}",
            f"{loc}.weight",
            kind="out_of_range",
        )
    _reject_bool_and_int(tx.fee, "fee", f"{loc}.fee")
    if tx.fee < 0:
        raise ValidationError(
            f"fee must be non-negative; got {tx.fee}", f"{loc}.fee"
        )
    if tx.fee > FEE_LIMIT:
        raise ValidationError(
            f"fee {tx.fee} exceeds limit {FEE_LIMIT}",
            f"{loc}.fee",
            kind="out_of_range",
        )
    inputs = _validate_str_tuple(tx.inputs, "inputs", f"{loc}.inputs")
    depends = _validate_str_tuple(tx.depends, "depends", f"{loc}.depends")
    if len(inputs) > MAX_INPUTS:
        raise ValidationError(
            f"inputs length {len(inputs)} exceeds limit {MAX_INPUTS}",
            f"{loc}.inputs",
            kind="out_of_range",
        )
    if tx.txid in depends:
        raise ValidationError(
            f"transaction {tx.txid!r} must not depend on itself",
            f"{loc}.depends",
            kind="cycle",
        )
    for op in inputs:
        idx = op.rfind(":")
        if idx > 0 and op[:idx] == tx.txid:
            raise ValidationError(
                f"transaction {tx.txid!r} must not spend its own output "
                f"{op!r}",
                f"{loc}.inputs",
                kind="cycle",
            )
    # Normalize via a rebuilt instance only when caller used lists.
    if inputs != tx.inputs or depends != tx.depends:
        tx.inputs = inputs
        tx.depends = depends


def make_tx(txid, weight, fee, inputs=(), depends=()):
    """Validate fields and return a canonical :class:`Tx`."""
    tx = Tx(txid, weight, fee, inputs, depends)
    validate_tx(tx)
    return tx
