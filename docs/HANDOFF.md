# 交接说明（Handoff）

目标读者：接管本项目的人（或三个月后的自己）。**不复制正文，只给指针**——细节在各自文档里。

最后更新：2026-11-04 ｜ 分支：`master` ｜ 基线 commit：`9f7f3e3`

> ## ✅ 已修复：数据层曾未被版本控制
>
> **问题**：`.gitignore` 里有一条未锚定的 `data/` 规则，git 在任何深度都会应用它，于是 `src/stockpred/data/` 被静默排除在六次提交之外。而引用它的 6 个测试模块已经进库 → **从 HEAD 新克隆的仓库连 collection 都过不去**：
>
> ```
> ModuleNotFoundError: No module named 'stockpred.data'
> ```
>
> **这一条是通过「克隆 HEAD 再跑测试」发现的，不是通过读代码发现的。**
>
> **修复**（commit `9f7f3e3`）：`.gitignore` 改为锚定的 `/runs/`、`/data/`、`/logs/`，并补交整个数据层。
>
> **修复过程中又踩了一次同类坑**：`data/__init__.py` 会 re-export `crosscheck`，而我把 `crosscheck.py` 当作「别人的在建文件」排除，结果包仍然不可导入。克隆验证第二次抓到，已补入。
>
> **验收标准（建议每次改动后都跑）**：
>
> ```bash
> git clone . /tmp/cc && cd /tmp/cc
> PYTHONPATH=src python -c "import stockpred.data.schema, stockpred.pipeline"
> ```
>
> 现在 10 个顶层模块从干净克隆全部可导入（已验证）。


## 1. 三十秒版本

`Stock-Prediction-Models` 的旧 notebook（TF1，2021 年）旁边新建了一个 Python 包 `src/stockpred`，把「模型 zoo」补上了工程脊柱：数据层、含 A 股规则的、无泄漏的评估、决策链、风控硬门、纸面撮合与对账、服务契约、合规报备材料、监控与审计链。

**没有动任何旧 notebook 与数据文件。** 旧代码仍是可读教材，新代码是能跑的系统。

**它现在能做什么**：给定行情，产出可复盘的决策包与委托单，按 A 股规则纸面撮合并对账。
**它现在不能做什么**：接真实行情、向券商下单。前者是待办，后者阻塞于券商书面答复。

## 2. 先读什么（按顺序，约 20 分钟）

| # | 文档 | 为什么先读它 |
|---|---|---|
| 1 | [BRIEF.md](../BRIEF.md) | 目标 / 非目标 / 已确认决策 / 验证命令 |
| 2 | [MIGRATION-REPORT.md](MIGRATION-REPORT.md) | 迁移了什么、**故意没迁什么**、为什么 |
| 3 | [EVAL-PROTOCOL.md](EVAL-PROTOCOL.md) | 任何策略数字必须遵守的协议（含口径定义） |
| 4 | [DECISION-PIPELINE.md](DECISION-PIPELINE.md) | 决策链三段式 + 五项必填字段 + 风控门 |
| 5 | [TECH-DEBT.md](TECH-DEBT.md) | 按「会不会伤到钱」排序的未决项 |

## 3. 代码地图

```
src/stockpred/
├── data/         行情获取、schema 校验、内容寻址缓存、交易日历、陈旧/前视守卫
├── backtest/     A 股成本模型、T+1/整手/涨跌停规则、逐日撮合引擎
├── eval/         walk-forward + purge/embargo、泄漏检测、指标、报告、三重障碍标注
├── models/       统计基线、neuralforecast 适配、模型对比
├── agents/       8 个迁移后的 agent（DQN 家族 / AC 家族 / 演化策略）
├── signals/      4 个规则型信号（turtle / MA / signal-rolling / abcd）
├── envs/         Gymnasium 环境（含 legacy 兼容模式）
├── universe/     PIT 标的池过滤
├── portfolio/    权重构建 + 100 股整手落地
├── risk/         风控硬门 + 换手预算
├── execution/    委托单、纸面撮合、对账
├── monitor/      哈希链审计 + 漂移检测
├── serving/      决策 API（契约层 + stdlib 传输层）
├── compliance/   程序化交易报备材料生成
├── tune/         optuna 搜索（可续跑 study）
├── pipeline.py   alpha → portfolio → execution → 风险门 → 决策包
├── config.py     Settings（含 broker_confirmed_at 硬门）
├── runs.py       run 记录（config 哈希）
└── errors.py     7 个类型化错误
```

规模：`src` 56 文件 / 7,730 行；`tests` 30 文件 / 4,218 行 / **415 用例**（含并行会话新增的 `test_crosscheck.py`）；`tools` 脚本 / `docs` 15 份。

## 4. 常用的四件事

```bash
# 1) 生成一个决策包（合成数据，验证管线通不通）
.venv/Scripts/python tools/demo_decide.py

# 2) 12 日纸面循环 + 对账 + 审计链校验
.venv/Scripts/python tools/demo_paper_loop.py 12

# 3) 含成本回测 + 三基准报告
.venv/Scripts/python tools/demo_backtest.py     # → docs/SAMPLE-REPORT.md

# 4) 生成程序化交易报备草稿
.venv/Scripts/python tools/filing.py --name 张三 --account A123456789 --broker 某证券
```

决策 API（本机回环）：

```bash
.venv/Scripts/python -c "from stockpred.serving.app import serve; serve()"   # :8005
curl http://127.0.0.1:8005/health
```

## 5. 当前门状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| S0 对齐 | ✅ 通过 | [BRIEF.md](../BRIEF.md) |
| S1 工程脊柱 + 数据层 | ✅ 通过 | 60 用例；许可证门红→绿四轮 |
| S2 可信评估（实盘前置） | ✅ 通过 | 成本对账 5.1 元；泄漏用例可红；滑点单调 |
| S3 决策链 | ✅ 通过 | 五项字段；`target_invested` vs `executed_notional` 分离 |
| S4 纸面灰度 | ✅ 通过 | 12/12 日对账一致；审计链 12 条完整 |
| S5 服务 + 合规 + 方法论 | ✅ 通过 | `/decide` 200；报备五类字段；三重障碍 + purge |
| S6 复盘 + 文档 | ✅ 通过 | 迁移报告 / 技术债 / 六类复盘 |
| S7 战略 spike | ✅ 通过 | TimesFM 未胜出；qlib 不整体接入 |
| **S6' 实盘放行** | ⛔ **未开始** | 阻塞：券商书面答复（`broker-written-inquiry`） |

## 6. 必须知道的六个陷阱（都是实际踩过的）

1. **空字典 ≠ 缺字段**。`risk: {}` 表示「没有拒单」，是完整答案；早期把空 dict 判为缺失导致决策包无法导出。
2. **目标金额 ≠ 已执行金额**。`target_invested` 与 `executed_notional` 必须分开，混用会把「想买」当成「已买」。
3. **组合上限 > 风控上限 = 静默全拒**。现在装配期直接抛错。
4. **报告格式不能靠数值大小猜**。曾把 Sharpe 2.41 渲染成 241.27%。按列语义格式化。
5. **假实现比没有更危险**。`sample_uniqueness` 初版返回常数，看起来对，实际无意义。
6. **中文断言不要看控制台**。Windows 代码页会把 UTF-8 显示成乱码；用字节级或 Python 内比较。

## 7. 环境注意

- Python **3.11.9**，虚拟环境在 `.venv/`（不入版本控制）。
- `pyqlib` **已临时安装但未进 `pyproject.toml`**，它把 `pyarrow` 从 26 降到 25、重装了 `python-dateutil`。当前 404 用例全绿，但这属环境漂移（TECH-DEBT P2-15）。
- 恢复精简环境：`pip uninstall pyqlib` → `pip install -e ".[dev]"`（pyarrow 需手动回到 26）。
- 可选 extra：`[data]`（akshare/baostock）、`[models]`（torch/neuralforecast）、`[tune]`（optuna）、`[rl]`（gymnasium/SB3）。

## 8. 回滚

每个阶段一个 commit，可独立回退：

```
8862fc8  agent 迁移（8 迁 + 4 改落点）
e842d06  RL 环境 + 迁移报告 + 两个 spike
e6efc46  决策 API + 报备材料 + 三重障碍
44de0d5  纸面撮合 + 对账 + 监控审计
eb4616b  决策链（alpha→portfolio→execution→风控）
4cdef6f  工程脊柱 + A 股回测 + 无泄漏评估
3326673  上游原始状态（fork 基线）
```

回退到基线：`git reset --hard 3326673` —— **这会丢失全部新增代码但旧 notebook 从未被修改**，因此 `dataset/`、`output/` 完好。

## 9. 未决项（按优先级）

| 优先级 | 项 | 谁来做 | 前置 |
|---|---|---|---|
| P0 | 券商书面答复（门槛口径 / miniQMT / 库白名单 / 谁报备 / 条件单定性） | **人类**（发邮件/工单） | 模板已备：[BROKER-INQUIRY.md](BROKER-INQUIRY.md) |
| P0 | 券商执行适配器 + 审批流 + 幂等 + kill switch | 工程 | 上一项 |
| P0 | 报备材料**实际提交**（生成器已有） | 人类 + 券商 | 同上 |
| P0 | 服务鉴权（上公网前） | 工程 | 无 |
| P1 | 真实行情端到端（akshare/baostock 联网实测） | 工程 | 无 |
| P1 | 目标仓位随权益再平衡（纸面 12 日后现金耗尽） | 工程 | 无 |
| P2 | CI 在 GitHub runner 实跑 / 类型检查 / Dockerfile | 工程 | 无 |

完整清单见 [TECH-DEBT.md](TECH-DEBT.md)。

## 10. 一句话建议给接手者

**先做「真实行情端到端」再谈任何策略结论。** 目前所有数字（含 12 日对账、模型对比、策略跑输买入持有）都来自固定 seed 的合成序列——机制已验证，结论未验证。这两者的区别，是本项目所有文档反复标注的同一件事。
