# Agent 迁移报告：8 迁 + 4 改落点 + 11 参考

`agent/` 目录共 **24** 个 notebook（编号 `1..23` + 未编号的 `updated-NES-google.ipynb`）。逐个核对类结构与实现后分三类处置，而不是「23 个一起搬」。

## 1. 分类依据（怎么判的）

对每个 notebook 提取了 `class` 定义、`replay(`/`self.memory`（是否为 DQN）、`Actor`/`Critic` 类、`Deep_Evolution_Strategy`/`NeuroEvolution` 类，以及 `window_size`：

| 特征 | 命中数 | 含义 |
|---|---|---|
| `window_size = 30` | 19 | 有学习/搜索过程，状态是 30 根价格差分 |
| 无 `window_size`、无 class | 4 | 纯规则信号（无状态、无动作） |
| `replay()` + `self.memory` | 2（#5、#10） | DQN 家族 |
| `Actor` + `Critic` | 4（#14–#17） | actor-critic 家族 |
| `Deep_Evolution_Strategy` / `NeuroEvolution` | 5 | 演化策略家族 |

## 2. 迁到 `AshareTradingEnv` 的 8 个

| 新实现 | 旧 notebook | 变体特征 |
|---|---|---|
| `DQNAgent` | [5.q-learning](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/5.q-learning-agent.ipynb) | 基线 DQN |
| `DuelingDQNAgent` | [10.duel-q](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/10.duel-q-learning-agent.ipynb) | dueling 头 |
| `DoubleDuelRecurrentDQNAgent` | [13.double-duel-recurrent-q](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/13.double-duel-recurrent-q-learning-agent.ipynb) | double + dueling + 循环编码 |
| `ActorCriticAgent` | [14.actor-critic](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/14.actor-critic-agent.ipynb) | softmax 策略 + 价值基线 |
| `ACDuelRecurrentAgent` | [17.actor-critic-duel-recurrent](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/17.actor-critic-duel-recurrent-agent.ipynb) | dueling logits + 循环编码 |
| `EvolutionStrategyAgent` | [6.evolution-strategy](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/6.evolution-strategy-agent.ipynb) + [updated-NES-google](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/updated-NES-google.ipynb) | 高斯扰动 + 秩归一化 |
| `NeuroEvolutionAgent` | [21.neuro-evolution](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/21.neuro-evolution-agent.ipynb) | 种群 + 锦标赛 + 变异/交叉 |
| `NoveltySearchESAgent` | [22.neuro-evolution-novelty-search](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/22.neuro-evolution-novelty-search-agent.ipynb) | ES + 行为新颖性档案 |

**框架**：TF1 → PyTorch（ES/NE 用 numpy，本就无需梯度）。

**关键点**：三个 DQN 变体在原始仓库里是**三份几乎相同的代码**，差别只在 double / dueling / recurrent 三个开关。迁移时保留为一个实现 + 三个命名类——照抄三份才是把原问题一起搬过来。AC 两个变体同理。

## 3. 改落点的 4 个（不进 env）

| 旧 notebook | 新落点 | 为什么不该进 env |
|---|---|---|
| [1.turtle](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/1.turtle-agent.ipynb) | `signals.TurtleSignal` | 唐奇安通道：无学习、无动作空间，输出一条 signal 向量 |
| [2.moving-average](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/2.moving-average-agent.ipynb) | `signals.MovingAverageSignal` | 双均线交叉，同上 |
| [3.signal-rolling](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/3.signal-rolling-agent.ipynb) | `signals.SignalRollingSignal` | 阈值 + 迟滞状态机，同上 |
| [23.abcd-strategy](c:/Users/Ricky/GitHub/Stock-Prediction-Models/agent/23.abcd-strategy-agent.ipynb) | `signals.ABCDSignal` | MA 形态判定，同上 |

它们实现 `AlphaModel` 协议（`score(bars, as_of) -> Series`），因此能直接塞进 `Pipeline`，与 `MomentumAlpha` 平级。塞进 env 属于范畴错误：env 是「学出来的策略」的容器，这四个是被规则写死的信号。

## 4. 保留为参考的 11 个

| notebook | 保留原因 |
|---|---|
| `7.double-q`、`11.double-duel-q`、`12.duel-recurrent-q` | double / dueling / recurrent 的其余开关组合，与已迁 3 个同族；独立理论价值低 |
| `8.recurrent-q`、`9.double-recurrent-q` | 同上 |
| `15.actor-critic-duel`、`16.actor-critic-recurrent` | 已迁 14（基线）与 17（全开关），中间组合不额外提供信息 |
| `18.curiosity-q`、`19.recurrent-curiosity-q`、`20.duel-curiosity-q` | 内在好奇心奖励在原实现中与回测收益混在一起，奖励口径未与 env 对齐；迁过来需要重做奖励设计，不是移植 |
| `updated-NES-google` | 内容已并入 `EvolutionStrategyAgent`（秩归一化 + 自适应步长），不单独保留实现 |

**这不是删掉它们**：notebook 原样保留，可读性是该仓库最不可替代的资产（见 [MIGRATION-REPORT](MIGRATION-REPORT.md) §4）。

## 5. 接口不兼容点（迁移的真实工作量）

| 维度 | 旧实现（19 个学习型） | 新 env |
|---|---|---|
| 观测 | 30 维价格差分（`block[i+1]-block[i]`） | `observation_mode="price_diffs", window_size=30` → 29 维差分 |
| 动作 | 3 个：买 1 单位 / 卖 1 单位 / 不动 | `action_mode="unit"` → `Discrete(3)`；默认 `lots` 模式为 `2*max_lots+1` |
| 奖励 | `0.7 × 平均单笔收益率 + 0.3 × 总资金收益率` | 权益变化 / 初始资金 |

前两项已通过 env 的两个适配模式解决（约 60 行），因此**不需要重写 agent 的输出层**。第三项有意不改：原奖励把「单笔收益率」混进来，会把「交易次数少但每次赚」的策略排到「整体赚更多」之前，与实盘目标不一致。

## 6. 用法

```python
from stockpred.envs import AshareTradingEnv
from stockpred.agents import EvolutionStrategyAgent

env = AshareTradingEnv(
    bars, max_lots=3, observation_mode="price_diffs", action_mode="unit", window_size=30
)
obs, _ = env.reset(seed=0)
agent = EvolutionStrategyAgent(obs.shape[0], env.n_actions)
stats = agent.learn(env, total_steps=200)
action = agent.act(obs, deterministic=True)
```

## 7. 验证

| 项 | 结果 |
|---|---|
| 8 个 agent 全部实现 `AgentPolicy` 协议 | 45 个用例（每个 agent × 6 项契约）通过 |
| 观测维度不匹配 | 抛 `AgentError`，不静默 reshape |
| 同 seed 贪心动作可复现 | 通过 |
| ES 权重确实移动 | 通过 |
| NE 种群规模守恒 + 精英保留 | 通过 |
| 新颖性档案随代数增长 | 通过 |
| DQN 回放缓冲区有界 | 通过 |
| AC 损失有限、熵 > 0 | 通过（曾出过 shape 广播 bug，已修） |

**未做**：没有在真实数据上把这些 agent 训出可用策略。它们现在证明的是「接口通了、能学」，不是「能赚钱」——后者需要 [EVAL-PROTOCOL](EVAL-PROTOCOL.md) 的完整流程与真实行情。
