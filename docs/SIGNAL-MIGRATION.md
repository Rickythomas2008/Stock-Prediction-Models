# 规则信号迁移：4 个 notebook agent → `stockpred.signals`

`agent/` 里 4 个 notebook 不是「学出来的策略」，而是写死的规则：没有状态空间、没有动作空间、没有训练。它们不该进 `envs`（见 [AGENT-MIGRATION](AGENT-MIGRATION.md) §3），而应作为信号生成器，与 `MomentumAlpha` 平级地喂给 `Pipeline`。

本文记录：每个信号的**原始定义**、**迁移后口径**、**与原实现的差异**，以及 ABCD 的复杂度替换。

| notebook | 新实现 | 分数范围 |
|---|---|---|
| `agent/1.turtle-agent.ipynb` | `signals.TurtleSignal` | −1 / 0 / +1 |
| `agent/2.moving-average-agent.ipynb` | `signals.MovingAverageSignal` | −1 / 0 / +1 |
| `agent/3.signal-rolling-agent.ipynb` | `signals.SignalRollingSignal` | 0 / +1（只做多） |
| `agent/23.abcd-strategy-agent.ipynb` | `signals.ABCDSignal` | −1 / 0 / +1 |

## 0. 共同约定

所有四个类实现 `stockpred.pipeline.AlphaModel`：

```python
name: str
def score(self, bars: Mapping[str, pd.DataFrame], *, as_of: date) -> pd.Series: ...
```

- **打分口径**：分数是 `-1..+1` 的**姿态**（负=看空），不是预期收益；`TopNConstructor` 只拿它排序，不做数值解释。
- **as_of 截断**：统一走 `pipeline._history_until`（`date <= as_of` 后再按 `date` 排序），与 `MomentumAlpha` 同一份实现，所以「未来行」不会被任何信号看到。index 位置不参与判断。
- **历史不足**：`_score_series` 返回 `NaN` → 该 symbol 跳过。不截短窗口硬算，因为窗口短一截会静默改变信号强度。
- **全部跳过**：返回空 `Series`（`dtype=float`），不抛异常。是否「无标的可打分」由下游决定——`Pipeline` 会由 constructor 抛 `PipelineError("no scores to construct from")` 暴露出来。这一点与 `MomentumAlpha`（直接抛）不同，是有意为之：信号本身不该规定调用方的失败策略。
- **单点打分**：notebook 产出整条 per-bar 向量；这里只回答「截至 `as_of` 那一根的姿态」。per-bar 版本留在测试的因果性探针里（`_prefix_feature`）。
- **纯函数优先**：能拆的状态机拆成纯函数（`signal_rolling_states`），便于用价格路径做单测，也便于回测审计。

## 1. TurtleSignal（唐奇安通道）

**原始定义**（`1.turtle-agent.ipynb`）：

```python
count = int(np.ceil(len(df) * 0.1))
signals['RollingMax'] = signals.trend.shift(1).rolling(count).max()
signals['RollingMin'] = signals.trend.shift(1).rolling(count).min()
signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1   # 突破上轨 → 做空
signals.loc[signals['RollingMin'] > signals.trend, 'signal'] =  1   # 跌破下轨 → 做多
```

**迁移后**：

```python
reference = close[-count-1:-1]        # shift(1) + rolling(count) 的等价写法
close[-1] >  max(reference) → +1.0
close[-1] <  min(reference) → -1.0
否则 0.0
```

| 维度 | 原实现 | 迁移后 | 说明 |
|---|---|---|---|
| 方向 | 突破上轨 `-1`、跌破下轨 `+1` | 突破上轨 `+1`、跌破下轨 `-1` | **符号翻转**：原 notebook 是均值回归读数，迁到统一「负=看空」口径后必须是趋势跟随读数。数值口径见 brief 与本文 §0 |
| 参考区间 | `shift(1).rolling(count)` | 同样排除当根 | 保留，否则每根都是自身高点，规则永不触发 |
| 窗口长度 | `ceil(0.1 * len(df))`（随样本长度变化） | `count=20` 固定、可配 | 自适应窗口让同一策略在历史长短不同的样本上不可比；固定窗口让结果可复现、可跨标的横比 |
| 输出 | 整条向量 | 单个分数 | 生产侧只需 `as_of` 那一根 |
| 不足历史 | 前 `count` 行为 `NaN`（向量里保留） | 整条 `NaN` → 跳过 symbol | 不足窗口不硬算 |

## 2. MovingAverageSignal（双均线交叉）

**原始定义**（`2.moving-average-agent.ipynb`）：

```python
short_window = int(0.025 * len(df))
long_window  = int(0.05  * len(df))
signals['short_ma'] = df['Close'].rolling(short_window, min_periods=1).mean()
signals['long_ma']  = df['Close'].rolling(long_window,  min_periods=1).mean()
signals['signal'][short_window:] = np.where(short_ma[short_window:] > long_ma[short_window:], 1.0, 0.0)
signals['positions'] = signals['signal'].diff()
```

**迁移后**：取两条 MA 的**最新值**比较，`short > long → +1.0`，`short < long → -1.0`，相等 `0.0`；不足 `long_window` 行 → 跳过。构造时校验 `short_window < long_window`，否则 `ValueError`。

| 维度 | 原实现 | 迁移后 | 说明 |
|---|---|---|---|
| 窗口 | 按样本长度取 2.5% / 5% | `short_window=10`、`long_window=30` 固定可配 | 同上：固定窗口保证可比与可复现 |
| warm-up | `min_periods=1`，半截 MA 也参与比较 | 必须满窗 | 半截 MA 对半截 MA 不叫交叉；而且 `min_periods=1` 会让前 29 行的比较退化成噪声 |
| 方向 | 只有 `0/1`（只做多），另用 `diff()` 生成买卖点 | `+1 / -1 / 0` | 交叉本身是对称的，缺空头读数会让 `TopNConstructor` 无法把「最差」和「持平」区分开 |
| 参数校验 | 无 | `short_window < long_window` 强制 | 反转窗口等于换了个策略，不报错就会静默产生垃圾信号 |

## 3. SignalRollingSignal（阈值 + 迟滞）

**原始定义**（`3.signal-rolling-agent.ipynb` 的 `buy_stock`，去掉下单/记账部分）：

```python
current_val = real_movement[0]
state = initial_state                      # 1 = 持仓, 0 = 空仓
for i in range(1, n):
    if real_movement[i] < current_val and state == 0:
        if current_decision < delay_change_decision: current_decision += 1
        else: state = 1; current_decision = 0          # 价格下跌 → 买入
    if real_movement[i] > current_val and state == 1:
        if current_decision < delay_change_decision: current_decision += 1
        else: state = 0; current_decision = 0          # 价格上涨 → 卖出
    current_val = real_movement[i]
```

**迁移后**（纯函数 `signal_rolling_states`）：

```python
deviation = close[i] / mean(close[i-window+1 : i+1]) - 1
state == 0 且 deviation >  threshold → 想进 1
state == 1 且 deviation < -threshold → 想回 0
翻转后 delay 根 bar 内拒绝再次翻转
返回 0/1，score = float(state)
```

| 维度 | 原实现 | 迁移后 | 说明 |
|---|---|---|---|
| 触发量 | 与**上一根**价格比较（涨/跌） | 与 **window 根均线**的偏离度比较（阈值） | 单根比较把噪声当信号；偏离度 + 阈值才可调、可跨标的横比 |
| 方向 | 跌 → 买、涨 → 卖（逆势） | 高于均线 + 阈值 → 进；低于 −阈值 → 出（顺势） | 口径与 brief 一致；原 notebook 的「跌买涨卖」是它自己的回测结论，不是通用规则 |
| `delay` 语义 | 连续反向 bar 计数到 `delay` 才允许切换（逐根累加） | 距上次翻转**至少** `delay` 根 | 原语义在「反向一根、顺向一根」交替时永远不会切换；改成最小驻留期，迟滞才真的起作用 |
| 输出 | `0/1` 状态 + 成交明细 | `0 / +1` | 只做多保留：造 `-1` 等于凭空加一条原策略从未持有的空腿 |
| 副作用 | 打印成交、改现金、算收益 | 无（纯函数 + 纯打分） | notebook 把「规则」和「玩具券商」焊死，分离后才能审计 |

## 4. ABCDSignal（A-B-C-D 形态）

**原始定义**（`23.abcd-strategy-agent.ipynb`）：

```python
def abcd(trend, skip_loop=4, ma=7):
    ma = pd.Series(trend).rolling(ma).mean().values
    x = []
    for a in range(ma.shape[0]):
        for b in range(a, ma.shape[0], skip_loop):
            for c in range(b, ma.shape[0], skip_loop):
                for d in range(c, ma.shape[0], skip_loop):
                    if ma[b] > ma[a] and (ma[c] < ma[b] and ma[c] > ma[a]) and ma[d] > ma[b]:
                        x.append([a, b, c, d])
    ac_set = set(...); bd_set = set(...)
    signal[list(ac_set - bd_set)] = 1.0     # 买点
    signal[list(bd_set - ac_set)] = -1.0    # 卖点
```

**迁移后**：在 `MA(ma_window)` 上取最近 `lookback` 根，用**单趟摆动点检测**找交替的 high/low 转折，再取最近一个合法 A-B-C 三元组（A 为低、B 为高、C 为低，要求 `B>A`、`C<B`、`C>A`，且 A→B、B→C 的跨度 ≥ `skip_loop`），最后对照**最新一根**判定：

| 情形 | 分数 |
|---|---|
| 已跌破 C（`close < C`） | `-1.0` |
| C 之后出现过高于 B 的高点（D 段确认上行） | `+1.0` |
| A-B-C 成立但 D 尚未越过 B | `0.0` |
| 找不出 A-B-C（单调走势等） | `0.0` |
| 历史不足 `ma_window + 5` 根 | 跳过 symbol |

先判「跌破 C」再判 D：一旦跌破 C，这个「上-下-上」的读法就作废，无论 D 之前越没越过 B。

### 与原 O(n⁴) 版本的差异

1. **复杂度**：原版对全长历史做四层循环（`range(a, n, skip)` 三层嵌套 + 外层 `a`），量级 O(n⁴)，并把每个命中的四元组存进 Python list；n=1000、skip=4 时是千万到亿级迭代，所以 notebook 里要加 `%%time`，也无法放进决策循环。迁移版是**单趟摆动点检测 O(n) + 最近三元组回溯 O(#pivots) + 尾部极值 O(n)**，在固定窗口内常数代价。实测见 §5。
2. **输出语义**：原版返回整条向量，把「出现在 A/C 但不在 B/D」的 bar 标 `+1`、反之标 `-1`——这是**事后标记形态发生在哪**，必须等形态走完（含未来 bar）才能确定，因此不可交易。迁移版返回 `as_of` 时刻的**姿态**：D 是否已确认、形态是否已失败。
3. **窗口与步长**：原版看全历史，`skip_loop` 是搜索步长（采样稀疏化）；迁移版用 `lookback` 截断（保证单标的代价恒定、且不依赖更早的历史），`skip_loop` 改写为**最小腿长**——原版用步长跳过的短腿本来就不会被采样。
4. **形态重合的选择**：原版把所有命中四元组的下标并入集合（`ac_set`/`bd_set`），互相覆盖；迁移版显式只取**最近一个** A-B-C，语义单一，可解释。
5. **无形态**：两边都给 0（原版无四元组时 `signal` 全零）。

### 已知偏差（刻意保留）

- 摆动点检测用「方向反转」定义转折，MA 上的转折位置与原版离散采样点不会逐一重合；判断的是同一类形态，不是同一个下标。
- 末端若已形成更低的低点，回溯会取到上一个完整 A-B-C 三元组，再按「跌破 C」给 `-1.0`；这比原版集合覆盖更保守（少给买点）。

## 5. 性能（本机实测）

`ABCDSignal(lookback=60)._score_series` 单标的耗时（`time.perf_counter`，本机记录，2026-11-03）：

| 实现 | 单标的耗时 |
|---|---|
| 原 `abcd()`（O(n⁴)，n=500、skip=4 → **42,672,000** 个候选四元组） | 决策循环不可用 |
| `ABCDSignal._score_series`，n=500 | 中位 **0.100 ms**，p90 0.159 ms，max 0.764 ms（100 次） |
| `ABCDSignal._score_series`，n=5000 | 中位 **0.169 ms**（50 次） |

原版没有可比的实测数字：它的耗时随历史长度四次方增长（n=500 已是 4.3×10⁷ 个候选，n=5000 再放大约 10⁴ 倍）；迁移版的耗时只随 `lookback` 变化，与历史长度无关——n 从 500 涨到 5000 只多了 `rolling` 的向量化开销（0.07 ms）。测试阈值留足余量：`tests/test_signals.py::test_abcd_single_symbol_is_fast` 断言中位数 < 5 ms（约 50 倍余量）。

## 6. 测试地图（`tests/test_signals.py`）

| 断言 | 用例 |
|---|---|
| 方向：涨 → 正、跌 → 负/0 | `test_turtle_scores_breakout_up_and_breakdown_down`、`test_moving_average_follows_the_cross`、`test_signal_rolling_stays_long_only` |
| 不足历史 → 跳过 symbol、不抛错 | `test_symbol_with_short_history_is_skipped[*]`、`test_no_usable_symbol_returns_empty_not_an_exception[*]` |
| MA 参数校验 | `test_moving_average_rejects_inverted_windows` |
| 迟滞（0→1 后 `delay` 内不翻转） | `test_signal_rolling_states_respect_the_delay`、`test_signal_rolling_delay_is_what_blocks_the_flip`、`test_signal_rolling_states_are_prefix_stable` |
| ABCD 形态（D 确认 / D 未确认 / 跌破 C / 无形态） | `test_abcd_confirmed_d_scores_long`、`test_abcd_structure_without_d_confirmation_is_flat`、`test_abcd_break_below_c_scores_short`、`test_abcd_without_structure_is_flat` |
| ABCD 性能 | `test_abcd_single_symbol_is_fast` |
| 协议 + 可进 `Pipeline` | `test_signal_satisfies_the_alpha_protocol[*]`、`test_signal_drives_the_pipeline[*]` |
| 因果性（lookahead） | `test_scores_are_causal[*]`、`test_causality_guard_would_catch_a_leak`、`test_rows_after_as_of_are_ignored[*]` |

**因果性探针的做法**：把 `score` 拆成 `_score_series(close) -> float`，测试把「前 i 根」逐个喂进去组成 per-bar 特征，再交给 `eval.leakage.assert_causal`。同时断言该特征不是常值——否则因果性检查会因为「全 0」而空过。`test_causality_guard_would_catch_a_leak` 是反向对照，证明这个检查本身能红。

**`AlphaModel` 的 `isinstance`**：`pipeline.AlphaModel` 未标注 `@runtime_checkable`，直接 `isinstance` 会抛 `TypeError`；`pipeline.py` 不在本次改动范围内，因此测试模块在导入时调用 `runtime_checkable(AlphaModel)` 打开运行期标记（只翻转一个类属性，不改契约）。建议后续在 `pipeline.py` 上补 `@runtime_checkable` 后删除这行。
