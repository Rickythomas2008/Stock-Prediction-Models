# 迁移报告（S6 复盘）

对标对象：2026-11-03 观测的 20 个高星同类项目。本文回答三个问题——**哪些能力迁了、怎么迁的、哪些故意不迁**。

## 1. 起点与终点

| 维度 | 起点（2021 年 fork） | 终点（本次迁移后） |
|---|---|---|
| 代码形态 | 62 个 notebook（12.5 MB），8 个散落 .py | `src/stockpred` 46 个模块 / **6,243 行**（另有 `tools/` 7 个脚本 / 1,175 行） |
| 依赖声明 | 无（无 requirements / pyproject） | `pyproject.toml`，core 5 个依赖 + 3 个可选 extra |
| 测试 | 0 | 28 个测试文件 / **3,625 行** / **322 个用例** |
| 可运行性 | TF1 API，现代 Python 跑不动 | Python 3.11 + pandas 3.1 + torch 2.15 CPU |
| 数据层 | 2019 年静态 CSV | akshare/baostock 适配 + parquet 内容寻址缓存 + 交易日历 + 陈旧/前视守卫 |
| 评估 | 单次切分 | walk-forward + purge/embargo + 泄漏检测 + 三基准 + 净/毛双数字 |
| 交易规则 | 无（1 单位买卖） | T+1 / 100 股 / 涨跌停 / 停牌 / 佣金最低 5 元 / 印花税 / 过户费 |
| 风控 | 无 | 8 类硬门 + 换手预算 + 回撤熔断 |
| 留痕 | 无 | 五项字段决策包 + JSONL + 哈希链审计 |
| 服务 | 一个 Flask demo | 契约化决策 API（含旧 `/trade` 兼容层） |
| 合规 | 无 | 报备材料生成器 + 券商确认门 |

**没有删改任何原有 notebook、`dataset/`、`output/`**——fork 的对照价值是资产，迁移全部落在新目录。

## 2. 能力迁移矩阵

图例：**✅ 已实现为代码** ｜ **📦 采用为依赖** ｜ **📖 只借鉴设计** ｜ **⛔ 明确不迁**

| 能力 | 同类来源 | 迁移方式 | 本项目落点 |
|---|---|---|---|
| 可安装包 + 环境锁定 | 惯例（~17/20 有） | ✅ | `pyproject.toml`、src-layout |
| 依赖许可证白名单 | 惯例 | ✅ | `tools/check_licenses.py`（AGPL/GPL/Commons Clause 直接失败） |
| 数据获取 + 缓存 + schema | `yfinance`/`akshare` | 📦 + ✅ | `data/providers/*`、`data/cache.py`、`data/schema.py` |
| 交易日历与新鲜度守卫 | qlib/Lean 思路 | ✅ | `data/calendar.py`、`data/freshness.py` |
| 含成本回测 + A 股撮合规则 | Lean/backtrader/freqtrade | 📖 + ✅ 自研 | `backtest/{costs,rules,engine}.py` |
| Walk-forward + purge/embargo | neuralforecast `cross_validation` | ✅ 自研 | `eval/walkforward.py` |
| 未来函数检测 | 无同类现成方案 | ✅ 自研（按构造成立） | `eval/leakage.py` |
| 三重障碍 + meta-labeling | mlfinlab（已闭源） | ✅ 自研 | `eval/labeling.py` |
| 绩效指标 + 三基准 | backtrader analyzers / vectorbt stats | ✅ 自研 | `eval/{metrics,report}.py` |
| 统计基线 | statsforecast | ✅ 自研（零依赖） | `models/baselines.py` |
| 现代 DL 模型 | neuralforecast | 📦 | `models/neural.py`（NHITS/LSTM/PatchTST） |
| 模型对比同协议 | 无 | ✅ | `models/compare.py` |
| 超参搜索 + 可续跑 | optuna | 📦 | `tune/__init__.py`（SQLite study） |
| 组合构建 + 整手落地 | PyPortfolioOpt | ✅ 自研（含 100 股取整） | `portfolio/__init__.py` |
| 标的池 PIT 过滤 | qlib universe 概念 | ✅ | `universe/__init__.py` |
| 风控硬门 | 自研（Lean 组合约束思路） | ✅ | `risk/__init__.py`、`risk/turnover.py` |
| 三段式 alpha→portfolio→execution | Lean 框架模式 | 📖 + ✅ | `pipeline.py` |
| 委托单 + 纸面撮合 + 对账 | freqtrade dry-run 思路 | ✅ 自研 | `execution/{proposal,paper,reconcile}.py` |
| 审计链 + 漂移监控 | 无同类现成方案 | ✅ 自研 | `monitor/{audit,drift}.py` |
| 决策 API | OpenBB typed client 模式 | ✅ 自研（标准库） | `serving/{contract,app}.py` |
| 程序化交易报备 | 监管要求（CSRC 2024 No.8） | ✅ 自研 | `compliance/__init__.py` |
| RL 环境 API | FinRL / SB3 | 📦 + ✅ | `envs/__init__.py`（Gymnasium） |

## 3. 明确不迁移（含原因）

| 项 | 星数 | 不迁原因 |
|---|---|---|
| `QuantConnect/Lean` | ~22.4k | C#/.NET 重平台；只借 fee/slippage 与三段式**设计** |
| `mementum/backtrader` | ~23.6k | 2023-04 起停更 + GPL-3.0（传染）；只作 API 对照 |
| `hudson-and-thames/mlfinlab` | ~5.0k | **全权保留（闭源付费）**，代码已从仓库剥离；概念自研 |
| `jaungiers/LSTM-...` | ~5.2k | AGPL-3.0 + 2019 停更 |
| `polakowo/vectorbt` | ~9.7k | Apache-2.0 **+ Commons Clause**（禁止转售），非 OSI 开源 |
| `kernc/backtesting.py` | ~9.2k | AGPL-3.0（网络使用传染） |
| `firmai/financial-machine-learning` | ~8.8k | 无许可证文件的清单仓库，非软件 |
| 旧 repo 的 18 个 TF1 模型 | — | 不逐一重写为 PyTorch；由 neuralforecast + 1 个参考实现覆盖，notebook 保留为可读教材 |

## 4. 关键取舍（写下来，因为都是可争议的）

1. **不做「18 个模型全重写」**。同类里没有第二个项目做这么全的可读实现集合，这是原仓库唯一不可替代的资产；重写会把它变成平庸的第三个 FinRL。
2. **回测引擎自研而非引库**。要的 A 股规则（T+1 / 100 股 / 涨跌停 / 最低佣金）在 MIT/Apache 白名单里没有现成实现，AGPL 的 `backtesting.py` 不能用；自研约 250 行，换来可测与可改。
3. **服务用标准库而非 FastAPI**。契约层是纯函数，传输层只有 100 行；不为此引入 ASGI 生态。
4. **纸面撮合与回测引擎分开**。前者验证「订单能不能成」，后者回答「策略赚不赚」。混在一起会让两个问题互相掩盖。
5. **决策包必填五项字段**。少一项就无法事后复盘，而复盘是实盘系统唯一的学习机制。

## 5. S6 出口门

- [x] 能力对照表回填完成（本文 §2/§3）
- [x] 「不迁移」项逐条有原因（§3）
- [x] 技术债登记（[TECH-DEBT.md](TECH-DEBT.md)）
- [x] README 导航指向全部文档（避免文档漂在目录里没人看）
- [x] 322 用例全绿 / ruff 全绿 / 许可证门 PASS
