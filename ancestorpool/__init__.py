"""可复用的“祖先包约束交易池替换”题库。

公开接口见模块文档字符串；本实现只覆盖本题契约定义的有限数学/协议子集，
不宣称兼容任何完整行业标准（如完整 BIP125 RBF 规则）。
"""

from .errors import (
    AncestorPoolError,
    InvalidArgumentError,
    InvalidTransactionError,
    DuplicateTransactionIdError,
    UnknownDependencyError,
    PoolFullError,
    ReplacementRejectedError,
    SearchBudgetExhaustedError,
    NonFiniteNumberError,
)
from .model import Outpoint, Transaction, MAX_TRANSACTIONS
from .pool import TxPool, AddResult
from .solver import SelectionResult, Exclusion, DEFAULT_NODE_BUDGET

__all__ = [
    "Outpoint",
    "Transaction",
    "TxPool",
    "AddResult",
    "SelectionResult",
    "Exclusion",
    "MAX_TRANSACTIONS",
    "DEFAULT_NODE_BUDGET",
    "AncestorPoolError",
    "InvalidArgumentError",
    "InvalidTransactionError",
    "DuplicateTransactionIdError",
    "UnknownDependencyError",
    "PoolFullError",
    "ReplacementRejectedError",
    "SearchBudgetExhaustedError",
    "NonFiniteNumberError",
]
