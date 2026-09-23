# ancestorpool — 祖先包约束的交易池替换（离线题库）

一个只依赖 Python 标准库的可复用模块，实现本题契约下的两件事：

1. **`TxPool.add(tx)`**：向交易池加入合成交易；当新交易与池中交易消耗相同
   outpoint 时，按固定策略 *原子地* 替换“冲突交易及其所有后代”。
2. **`TxPool.select(block_weight)`**：在当前池上求块重约束内 **总手续费最大**
   的 **依赖闭合** 子集（祖先包），输出入选交易拓扑序与每个排除交易的理由。

这是本题自定义的有限数学/协议子集，**不宣称兼容任何完整行业标准**
（不是完整 BIP125 RBF，也不校验脚本、签名、时间锁等真实比特币规则）。

## 运行环境

- Python **3.14.7**（3.10+ 亦可运行，未使用 3.14 专有特性）
- **零第三方依赖**，仅标准库；无需网络、账号、密钥、数据库、中间件、Docker。
- Linux 原生运行：

```bash
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 约 0.1 秒内完成的真实计算演示
```

## 公开接口

```python
from ancestorpool import (
    Outpoint, Transaction, TxPool,
    AddResult, SelectionResult, Exclusion,
    MAX_TRANSACTIONS, DEFAULT_NODE_BUDGET,
    # 异常
    AncestorPoolError, InvalidArgumentError, NonFiniteNumberError,
    InvalidTransactionError, DuplicateTransactionIdError,
    UnknownDependencyError, PoolFullError,
    ReplacementRejectedError, SearchBudgetExhaustedError,
)
```

### 数据结构

- `Outpoint(txid: str, vout: int)`：被消耗的输出点，冻结值对象。
  `vout` 为非负整数（拒绝 `bool`）。
- `Transaction(txid, weight, fee, inputs=(), depends_on=())`：
  - `txid`：非空 `str`，池内唯一；
  - `weight`：**正整数**；`fee`：**非负整数**；
  - `inputs: Iterable[Outpoint]`：本交易消耗的 outpoint 集合（去重）；
  - `depends_on: Iterable[str]`：显式父交易 txid；选中本交易时这些祖先
    （及其祖先闭包）必须同时入选。
- `TxPool()`：内存交易池。方法：
  - `add(tx) -> AddResult`；`select(block_weight, node_budget=1_000_000) -> SelectionResult`；
  - `orphans() -> dict[str, tuple[str,...]]`：观测替换后孤儿；
  - `transactions()`、`get(txid)`、`size()`、`len(pool)`、`txid in pool`。
- `SelectionResult` 字段：`selected`（拓扑序元组）、`total_fee`、`total_weight`、
  `block_weight`、`exclusions`、`nodes_visited`、`proven_optimal`。

### 输入上限（接口边界）

- 池中交易 **最多 30 笔**（`MAX_TRANSACTIONS`）；`depends_on` 构成的图必须
  是 DAG（无环、无未知依赖、无自依赖）。
- `block_weight`、`node_budget` 均为正整数（拒绝 `bool`）。
- 短演示不穷举上限规模。

### 替换规则（`add`）

设冲突集合 `C` = 池中与新交易消耗同一 outpoint 的交易，
待移除集合 `R` = `C` 中每笔交易沿 `depends_on` 的全部后代（含自身）。

按下列顺序判定，任一不满足即拒绝且 **池状态不变**：

1. `txid` 已在池中 → `already_in_pool`；
2. 存在未知 `depends_on` → `UnknownDependencyError`（加入后成环同样拒绝）；
3. 新交易依赖 `R` 中交易（显式 `depends_on`，或消耗了 `R` 中交易创建的
   outpoint）→ `depends_on_removed`；
4. `tx.fee <= Σfee(R)` → `fee_not_strictly_greater`（**严格大于**）；
5. `tx.fee/tx.weight <= Σfee(R)/Σweight(R)` → `feerate_not_higher`
   （**严格高于**；整数交叉相乘，不产生浮点）；
6. 无冲突且池已满 30 笔 → `PoolFullError`（替换式加入移除数 ≥1，不会超限）。

接受时一次性删除 `R`、加入新交易（提交前不改动池，故失败原子、无部分变更）。

**替换后孤儿**：池中某交易消耗的 outpoint 其创建交易随替换被移除、且它没有
对创建者的显式 `depends_on`。本契约允许该状态残留并显式暴露：
`orphans()` 可查询，`select` 将孤儿根强制排除（理由
`missing_input_outpoint`），其 DAG 后代按祖先规则排除（`ancestor_excluded`）。

### 选择规则（`select`）

- 可行解：依赖闭合且 `Σweight <= block_weight`（空集恒可行）。
- 目标：`Σfee` 最大；**同分破平**：取“入选 txid 升序元组”字典序最小者
  （确定、与添加顺序无关）。`selected` 输出为拓扑序（父先子后，同层按 txid）。
- 算法：带祖先强制/否决传播的深度优先分支定界，含闭包超重剪枝与精确
  分数背包费用上界（全整数交叉相乘）。
- **搜索预算明确**：以搜索节点数计（`node_budget`，默认
  `DEFAULT_NODE_BUDGET = 1_000_000`）。预算耗尽抛
  `SearchBudgetExhaustedError`（带 `nodes_visited`/`node_budget`），
  含义是“**未能证明最优**”。本契约空集恒可行，因此不存在“已证明无可行解”
  的返回路径——预算耗尽与无解在类型与语义上严格分开。

### 排除理由代码

| 代码 | 含义 |
| --- | --- |
| `ancestor_closure_overweight` | 该交易连同全部祖先闭包总重超过块重，任何包都放不下 |
| `ancestor_excluded` | 祖先未入选（或被强制排除），依赖无法闭合 |
| `missing_input_outpoint` | 替换后孤儿：消耗的 outpoint 创建者已被移除 |
| `not_in_optimal_package` | 闭合可容纳，但块重内存在费用不更低、字典序不更差的包 |

## 错误语义总则

- 数值字段拒绝 `NaN`/`±Infinity`（`NonFiniteNumberError`）；整数参数拒绝
  `bool`（`True`/`False` 不会被当作 1/0）；数值字段只接受 `int`，传入
  `float` 一律类型错误。
- 非法输入抛出带可定位字段的异常（消息含 `[field=...]`、`[txid=...]`），
  不做静默纠正或默认值兜底。
- 有状态操作（`add`）在任何提交前完成全部校验，失败不留部分变更。

## 数值与误差口径

- `weight`、`fee`、`block_weight` 与所有费用/费率比较全部使用 **整数**：
  费率比较以 `fee1*w2 ? fee2*w1` 的交叉相乘完成。
- **本库不存在浮点运算结果，误差口径为 0**；不会用宽容差掩盖边界争议，
  严格大于/等号边界由整数精确定义（等号费率不通过）。

## 测试与独立参考

- `python -m unittest discover -s tests -v`
- `tests/reference.py` 是 **独立** 参考：枚举全部 2^n 子集（n ≤ 16）独立计算
  闭合最优与拓扑序，**不调用被测核心** 生成期望值；被测求解器与其对拍
  （含 120+ 随机 DAG、手工链/菱形、同分密集场景、16 节点上限用例）。
- 另覆盖：低费父高费子、替换后孤儿、间接冲突（后代链）、费用/费率严格性、
  失败原子性、池上限、预算耗尽与“有解可证”的区分、输入校验。

## 设计取舍

- 用 `depends_on` 显式表达 DAG 祖先关系；outpoint 双花只用于触发替换冲突，
  二者解耦，便于独立测试“间接冲突”和“无显式依赖的孤儿”。
- 选 30 笔上限是接口边界；分支定界在该规模下节点数很低（独立 0/1 背包
  最难情形约数千节点，链/DAG 通常数十节点），默认预算余量充足。
- 替换不级联删除“非 DAG 后代但双花相关”的交易，而是以孤儿理由暴露，
  使边界行为可观测、可测试。

## 已知限制

- 不是完整比特币内存池/CPFP/RBF 实现：无脚本、签名、mempool 一致性包、
  sigop、时间锁、硬币面额守恒等校验；outpoint 的“创建者”不要求在池中
  （外部输入），仅当创建者 *曾入池后被替换移除* 时才计为孤儿。
- 不持久化、不并发安全（单进程内存数据结构）。
- 搜索为精确最优；若人为设置极小预算，调用方需自行处理
  `SearchBudgetExhaustedError`（或调大预算），库不返回“近似最优”。

## 仓库结构

```
ancestorpool/          # 题库：errors/model/solver/pool
tests/                 # unittest 测试 + 独立穷举参考
demo.py                # 固定输入演示（正常结果 + 真实拒绝 + 真实预算异常）
requirements.txt       # 仅注释：标准库，无需安装
```
