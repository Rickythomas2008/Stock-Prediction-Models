# 决策链（S3）：alpha → 组合 → 执行 → 风控 → 决策包

## 1. 为什么是这条链

本地旧代码把「模型 → 买卖」直接连在一起：预测出信号就买 1 单位。实盘不可用，因为中间缺三层判断——**买多少**（组合）、**能不能买**（标的池与风控）、**买了留下什么证据**（决策包）。

```
bars ──▶ AlphaModel ──▶ PortfolioConstructor ──▶ to_lots ──▶ plan_execution ──▶ RiskGate
              │                    │                            │                  │
         横截面打分           目标权重 + 100 股取整          目标 vs 持仓差分        硬门
                                                                       │
                                                            DecisionPackage（五项必填）
```

## 2. 五个必填字段（缺一项即拒绝输出）

`REQUIRED_DECISION_FIELDS = as_of, bar_date, signal, target_shares, risk, rationale`

- `signal` 与 `target_shares` 必须非空；
- `risk` **允许为空字典**——「无拒单」是完整答案，不是缺字段（这条曾经写错，把空 risk 当缺失，已被测试锁死）；
- `bar_date > as_of` 直接抛 `PipelineError`（未来 bar）。

另外两个金额字段不可混用：

| 字段 | 语义 |
|---|---|
| `target_invested` | 组合层想要投多少（可能被拒单打折） |
| `executed_notional` | 风控实际放行的金额 |
| `cash_after` | 执行后的现金 |

三者不同源，混用会把「目标」当成「已建仓」。

## 3. 三个阶段的替换契约

| 阶段 | 协议 | 默认实现 |
|---|---|---|
| alpha | `score(bars, as_of) -> pd.Series` | `MomentumAlpha(lookback)`：close / 20 日均值 − 1 |
| 组合 | `construct(scores, returns, capital) -> pd.Series` | `TopNConstructor(top_n, max_position)` / `EqualWeightConstructor()` |
| 执行 | `plan_execution(target, state, prices, gate)` | 先卖后买、按差值生成限价单 |

`Pipeline` 会在**装配时**校验组合层 `max_position` ≤ 风控 `max_position_pct`。不校验的后果是所有单子被静默拒绝，管线看起来「跑通」但从不交易。

## 4. 风控门（硬门，不信任模型）

`RiskGate.evaluate(proposal, state)` 返回 `GateResult(approved, violations)`，违反项可枚举：

`max_orders_per_day` / `single_order_too_large` / `position_limit` / `gross_exposure_limit` / `insufficient_cash` / `no_position_to_sell` / `drawdown_circuit_breaker` / `turnover_budget_exceeded`

- 被拒的提案**仍计入**当日委托笔数（真实券商也这么算），但不改变持仓与现金；
- 回撤熔断基于 `peak_equity`，需要每日 `mark_to_market` 更新；
- 现金缓冲默认 5%，避免最后 100 股买不进而反复试单。

## 5. 换手预算（10 万本金的核心约束）

| 指标 | 实测 |
|---|---|
| 单笔 1 万元往返成本率 | **0.152%** |
| 每月 2 次往返的年化拖累 | **3.65%** |
| 5% 年化拖累上限内允许的月往返次数 | **2 次** |

`TurnoverBudget` 按月累计已用额度，`RiskGate` 在末端拒绝超预算提案。这条比任何预测精度都更决定 10 万账户的成败。

## 6. 标的池（PIT 正确性）

`build_universe(frame, as_of=...)` 只用 `date <= as_of` 的行，排除：停牌、ST、退市、封板、流动性不足（<500 万成交额）、单价过高（一手买不起）、历史过短。

两个泄漏用例已被测试锁死：
- 只在 `as_of` **之后**出现的新股，今天不能选；
- `as_of` **之后**才发生的退市，今天的样本不能「提前知道」。

## 7. 模型层

- 基线（零依赖）：naive / drift / moving_average / seasonal_naive
- 神经（可选 extra `[models]`）：`NeuralForecastAdapter` 包装 neuralforecast 的 `NHITS` / `LSTM` / `PatchTST`，CPU、固定种子、逐位可复现
- `compare_models()` 让基线与神经模型走**同一个** `_rolling_mae` 路径，输出同表；单模型失败记为 `mae=nan` 不拖垮整次比较

⚠️ 在 250 点合成序列、`max_steps=30` 下的排名（PatchTST/LSTM 领先）**不构成有效性证据**；正式结论必须在真实数据 + 含成本回测 + 更多 origins 上重跑。

## 8. 超参搜索

`run_search(objective, SearchSpace(...), storage="runs/tune.db")`：TPE 采样 + SQLite 持久化，**同名 study 会续跑而非重开**（`load_if_exists=True`），试验数累加。空间非法在搜索前就抛错，不浪费中途失败。

## 9. S3 出口门

- [x] 决策包含五项字段（缺项测试变红）
- [x] 超限 / 超换手 / 无持仓卖出提案被拒且留痕
- [x] 组合层与风控层上限不一致时装配即失败
- [x] 三个神经模型 + 四个基线同表可比
- [x] 标的池无前视泄漏（两个用例）
- [x] 决策日志 append-only（JSONL，两次追加两行）
