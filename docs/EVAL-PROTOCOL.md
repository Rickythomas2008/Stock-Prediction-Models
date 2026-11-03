# 评估协议（实盘前置门 S2）

一句话：**在 S2 出口门通过之前，任何策略数字都不允许进入决策链。**

## 1. 三条不可协商的规则

### R1 样本外、不重叠

- 用 `stockpred.eval.walk_forward_folds` 生成折；测试窗口永远在训练窗口之后。
- `purge` 去掉训练集末尾与测试集标签重叠的行；`embargo` 去掉测试集之后紧邻的训练行。
- `assert_no_overlap` 在 CI 里跑；泄漏直接失败，不靠人工检查。

```python
folds = walk_forward_folds(dates, train_size=250, test_size=20, purge=5, embargo=1)
assert_no_overlap(folds)
```

### R2 成本前后双数字

同一个策略必须跑两遍：`CostModel()`（真实费用）与零成本版本。两者差异就是成本吞噬。

- 依据：10 万本金单笔 1 万元往返 ≈ 0.15%；月换手 2 次 ≈ 年化 3.6% 成本。
- 门限（`failing_gates`）：成本吞噬毛收益 > 50% ⇒ 不可晋级（换手过高）。

### R3 必须带基准

`BenchmarkSuite.table(strategy)` 输出同表三基准：买入持有、现金、随机信号（固定 seed）。

- 现金基准回答「交易到底值不值」；
- 随机信号回答「策略是否只是运气」；
- 买入持有回答「是否跑输躺平」。

## 2. 未来函数检测（自动化）

`stockpred.eval.leakage.assert_causal(feature_fn, frame)`：把数据截断到某个 cut 点重算特征，与全量计算的前缀比对。**只要追加未来行改变了历史特征值，就判定泄漏。**

已覆盖的典型泄漏：
- `shift(-1)`（偷看次日）
- `rolling(center=True)`（居中窗口）
- 全序列统计量（`df["close"].max()` 当作特征）

CI 用例：`tests/test_leakage.py`（含 3 个必须变红的用例）。

## 3. 指标口径（避免同名不同义）

| 指标 | 口径 |
|---|---|
| total_return | 末值/首值 − 1 |
| cagr | 按 252 交易日/年折算；天数 < 2 返回 0 |
| annual_volatility | 日收益样本标准差（ddof=1）× √252 |
| sharpe | (日超额均值 / 日超额样本标准差) × √252；标准差为 0 时返回 0 |
| max_drawdown | 峰值到谷值最深跌幅，**负数** |
| calmar | cagr / abs(max_drawdown) |
| turnover | 累计成交金额 / 平均权益（成本代理，非年化） |

## 4. 报告要求

`render_report()` 生成的报告必须包含：
- 净/毛两行指标 + 成本吞噬百分比；
- 三基准同表；
- 折区间表；
- 五条免责披露（含「不构成投资建议」）。

## 5. 当前基线（A9）

`stockpred.models.baselines`：naive / drift / moving_average / seasonal_naive，
`evaluate_forecasters` 做滚动起点评估（只用历史拟合）。

任何神经网络模型（Sprint 3 的 neuralforecast 三模型）必须先**跑过这四个基线**才允许进决策链；
未跑赢基线的模型不进入 `pipeline`。

## 6. S2 出口门清单

- [ ] `pytest -q` 全绿，含 leakage / walkforward / metrics / report / backtest 用例
- [ ] 泄漏用例能变红（`shift(-1)` / 居中窗口 / 全序列统计量）
- [ ] 回测成本对账：单笔 1 万元买入 = 5.1 元
- [ ] 涨跌停 / T+1 / 100 股 / 停牌用例能变红
- [ ] 滑点单调性：滑点↑ ⇒ 净收益↓
- [ ] 报告含净毛双数字 + 三基准
