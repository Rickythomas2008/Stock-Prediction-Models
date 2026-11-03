# 纸面交易与对账（S4）

## 1. 这一层在防什么

S3 能产出决策包，但「决策包说买 700 股」和「券商真的成交 700 股」是两件事。中间会丢：价格没触及、涨停封板、T+1 卖不掉、现金不够、人工抄错。S4 的工作是把这条缝焊死——**先纸面跑通，再谈真钱**。

```
决策包 ──▶ 委托单(Sheet) ──▶ 纸面撮合(Paper) ──▶ 对账(Reconcile) ──▶ 哈希链审计(Audit)
             │                    │                    │                    │
        人工可执行            按 A 股规则成交        期望 vs 回报         不可改行
```

## 2. 委托单（R1/R2 路线）

`build_order_sheet(package, plan)` 产出 `OrderSheet`：

- 只列**风控放行**的单子；被拒单子单列并带原因；
- 字段固定：`as_of, symbol, side, shares, limit_price, notional, valid_until, risk_status, note`；
- `to_csv()` 供导出，`to_text()` 供人工照单在券商 App 键入。

这条路线**不属于程序化交易**（CSRC 2024 No.8 第 2 条：程序自动生成或下达指令）。因此无需报备——但前提是**没有程序替人下单**，这也是为什么 `require_broker_confirmation()` 把守在所有自动下单路径前面。

## 3. 纸面撮合

`PaperBroker.submit()` 只用 A 股规则成交，不猜：

| 条件 | 结果 |
|---|---|
| 无 bar / 停牌 | 拒单 `suspended` |
| 全天封涨停（low ≥ 涨停价） | 拒单 `limit_up` |
| 全天封跌停（high ≤ 跌停价） | 拒单 `limit_down` |
| 成交价劣于限价 | 拒单 `limit_price_not_reached` |
| 现金不够一手 | 拒单 `insufficient_cash`，够几手买几手 |
| 当日买入当日卖 | 拒单 `t_plus_1` |

费用用 `CostModel`（佣金万 2.5 且最低 5 元、印花税卖出 0.05%、过户费 0.001%、滑点 bps）。

**边界**：它是执行路径验证器，不是收益模拟器。策略收益请用 `backtest/engine.py`。混用会把「撮合逻辑对不对」和「策略赚不赚」两个问题搅在一起。

## 4. 对账

`reconcile(expected, actual)` 把差异分成四类，因为四类要四种处置：

| 类别 | 含义 | 该查什么 |
|---|---|---|
| `missing_fill` | 我们以为成交，券商没有 | 执行 bug / 日志丢失 |
| `unexpected_fill` | 券商成交了我们没下的单 | **控制失效**，最高优先级 |
| `quantity_mismatch` | 股数不一致 | 整手取整 / 部分成交处理 |
| `price_mismatch` | 价格差超容差 | 数据源或滑点假设 |

容差显式传参（默认价 0.01 元、股数 0）。`ReconcileAccumulator` 累积多日，`ok` 取全周期。

## 5. 审计链

`monitor/audit.py`：每条记录带 `seq`、`prev`（上一条 hash）、自身 `hash`。

- `append_record()` 追加；
- `verify_chain()` 重放，定位第一处断裂：`seq` 错位（删/插）、`prev` 不符（改）、payload 与 hash 不符（篡改）、JSON 损坏。

改一行即断链——这是「事后可复盘」的技术保证，不是承诺。

## 6. 漂移监控

`monitor/drift.py` 三个信号：

| 信号 | 判定 | 含义 |
|---|---|---|
| 特征 PSI | >0.25 major / >0.10 moderate | 输入分布移动（市场换挡） |
| 预测均值偏移 | >1σ moderate / >2σ major | 输出移动（模型或标准化问题） |
| 命中率 | <45% major；比基线降 10pt moderate | **唯一直接度量钱**的信号 |

PSI 分箱边界取自基线分位，保证「基线固定、只看当前怎么偏」。

## 7. 连续运行证据

`python tools/demo_paper_loop.py 12` 产出 [SAMPLE-PAPER-RUN.md](SAMPLE-PAPER-RUN.md)：

- 12 个交易日，每日决策 → 委托 → 撮合 → 对账；
- **12/12 天对账一致**；
- 审计链 12 条记录完整；
- 漂移检查：无漂移。

数据是固定 seed 的合成序列。它证明的是**机制成立**，不是策略有效。

## 8. S4 出口门

- [x] 委托单只含放行单子，拒单带原因，字段齐全可人工照单执行
- [x] 未取得券商确认时，任何自动下单路径抛 `BrokerNotConfirmed`
- [x] 纸面撮合覆盖 T+1 / 涨跌停 / 停牌 / 限价 / 现金不足五类拒单
- [x] 对账四类差异均可检出，容差可配
- [x] 审计链可检出篡改、删除、乱序、损坏
- [x] PSI / 预测偏移 / 命中率三类告警可触发
- [x] 连续 12 个交易日对账一致 + 链完整
