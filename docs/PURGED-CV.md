# Purged / Embargoed CV 与三重障碍标注（C3 设计）

`mlfinlab` 已闭源（2021-12 起仓库仅剩 issue 占位、代码被剥离），因此这一层是自研。概念（Lopez de Prado）是公开的，实现与验证由本项目负责。

## 1. 问题

普通 K 折或单次切分在时序上会泄漏：

- 训练集最后几行的**标签**要看到测试窗口的价格才能确定 → 训练看到了未来；
- 相邻测试折之间若不留间隔，边界样本同时属于训练与测试。

本地旧代码（`stacking/`、`deep-learning/`）就是单次切分，这是它结果不可信的第一原因。

## 2. 实现（已落地）

### 2.1 Purge

`walk_forward_folds(dates, train_size, test_size, purge=N, embargo=M)`
（`src/stockpred/eval/walkforward.py`）

- `purge=N`：从训练窗口**末尾**去掉 N 行，让标签兑现时间早于测试窗口开始；
- `embargo=M`：测试窗口之后跳过 M 行再作为下一折训练数据（序列自相关导致的反向泄漏）；
- `assert_no_overlap(folds)`：任何 `train_end >= test_start` 直接抛错，进 CI。

### 2.2 标注与 purge 窗口的绑定

`triple_barrier_labels(...)`（`src/stockpred/eval/labeling.py`）返回 `LabelResult`，
其中 `horizon` 就是**必须的** purge 长度。两者不许各写一个数字：

```python
labels = triple_barrier_labels(close, upper_pct=0.02, lower_pct=0.02, horizon=5)
folds = walk_forward_folds(dates, train_size=250, test_size=20, purge=labels.horizon)
assert_labels_honour_purge(labels, train_end=250, test_start=255)  # 不满足即抛错
```

三重障碍定义：

| 标签 | 含义 |
|---|---|
| `UPPER (+1)` | 先触及上障碍（止盈） |
| `LOWER (-1)` | 先触及下障碍（止损） |
| `VERTICAL (0)` | 到期未触及，按时间障碍结平 |

`volatility_scaled=True` 用滚动波动率缩放障碍宽度，使不同波动率区间的标签可比。

### 2.3 样本唯一性

重叠标签会虚增有效样本量。`sample_uniqueness(labels)` 按 Lopez de Prado 的口径计算每个标签跨度内的平均 `1/concurrency`：

- `horizon=1` → 全部 1.0（无重叠）；
- `horizon=5` → 首行约 0.46，说明「30 根 bar × 5 步」远不是 30 个独立样本。

指标必须连同这个数字一起报，否则「样本量 30、准确率 60%」是假的。

### 2.4 Meta-labeling

`meta_labels(primary, returns)`：主模型定方向，元模型只定**仓位大小**（1 = 方向对，0 = 方向错）。这样信号质量与仓位决策解耦，回撤更容易控制——对 10 万本金比追求命中率更实际。

## 3. 已验证的「能变红」

| 用例 | 断言 |
|---|---|
| `test_label_horizon_leaking_into_test_window_is_rejected` | `horizon=5`、`train_end=20`、`test_start=22` → 抛「leaks into the test window」 |
| `test_purged_split_passes` | `test_start=30` 时通过 |
| `test_inverted_split_is_rejected` | 训练在测试之后 → 抛错 |
| `test_uniqueness_equals_one_without_overlap` | `horizon=1` → 1.0 |
| `test_uniqueness_drops_with_overlap` | `horizon=5` → < 1.0 |
| 已有：`test_leaky_fold_is_detected` | 手工构造重叠折 → 抛错 |

## 4. 尚未做（明确边界）

- **分数阶差分（fractional differentiation）**：保留记忆同时获得平稳性，尚未实现；
- **meta-labeling 的仓位映射**（用概率定 size，而非二值）尚未接进 `portfolio`；
- **多标的联合标注**（横截面三重障碍）未实现，目前是单标的时序口径。

这三项不影响当前出口门，登记为后续 spike。

## 5. 与出口门的关系

S2 的评估门用 `purge`/`embargo` 已保证无泄漏；本节把「标注 → purge 长度」的耦合显式化，避免后续换 horizon 时忘了同步 purge——那正是最容易悄悄泄漏的地方。
