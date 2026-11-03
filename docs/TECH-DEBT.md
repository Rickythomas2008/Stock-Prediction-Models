# 技术债与未决项

按「会不会伤到钱」排序，而不是按工作量。

## P0 — 会影响真实资金（上线前必须解决）

| # | 债务 | 当前状态 | 为什么是 P0 |
|---|---|---|---|
| 1 | 无券商执行适配器 | `execution/` 只有委托单与纸面撮合 | 自动下单是已确认目标，缺这一层就无法落地 |
| 2 | 券商书面答复未取得 | `docs/BROKER-INQUIRY.md` 模板已备，`broker-written-inquiry` 阻塞 | CSRC 第 8 条「先报告后交易」，未确认即下单属违规 |
| 3 | 报备材料未实际提交 | `compliance/` 能生成五类字段 | 生成 ≠ 提交；券商确认是人工门 |
| 4 | 服务无鉴权 | `serving/` 设计为本机回环 | 上公网前必须加认证 + TLS |
| 5 | 对账的「券商回报」来源是自身 | `demo_paper_loop` 用 PaperBroker 的 fills | 真对账必须接券商对账单，否则等于自己对自己 |

## P1 — 会让结果不可信

| # | 债务 | 说明 |
|---|---|---|
| 6 | 无真实行情端到端 | akshare/baostock 适配层只有注入式单测，**未联网实测**；演示全用合成序列 |
| 7 | 策略未在真实数据上验证 | 所有「策略排名」来自合成随机游走，不构成有效性证据 |
| 8 | TopN 目标不随权益再平衡 | 纸面运行 12 日后现金降至 1,222 元，后续买入被现金缓冲拒；应改为按当前权益重算目标 |
| 9 | 分数阶差分未实现 | 平稳性与记忆的取舍仍用简单收益率 |
| 10 | meta-labeling 未接进仓位 | `eval/labeling.py` 能算标签，但没用它定 size |
| 11 | 横截面三重障碍未实现 | 目前是单标的时序口径 |
| 12 | 多标的对账键不足 | 键为 `(date, symbol, side)`，同一天同标的同方向多笔会判重复并抛错 |

## P2 — 工程质量

| # | 债务 | 说明 |
|---|---|---|
| 13 | CI 未在 GitHub runner 实跑 | 本地 Windows 全绿，`ubuntu-latest` 未验证；首次 push 后需补日志 |
| 14 | baostock 许可证未核实 | allowlist 标注 `UNVERIFIED`，上线前须核实或替换 |
| 15 | pyqlib 未进依赖 | spike 期间临时安装（引入 200+ 传递依赖），未写入 `pyproject.toml` |
| 16 | 无 `nbmake` notebook 测试 | 旧 notebook 仍未经 CI 验证（有意如此：TF1 环境与主环境冲突） |
| 17 | 无 Dockerfile | 部署形态未定 |
| 18 | 决策日志无轮转 | `logs/decisions.jsonl` 与 `runs/paper-demo/audit.jsonl` 会无限增长 |

## 环境偏差记录（S7 spike 造成）

`pip install pyqlib` 引入 **222 个包**，并降级 `pyarrow 26.0.1 → 25.0.1`、重装 `python-dateutil`。

- 全量测试仍全绿（已验证），但这属于**环境漂移**；
- qlib **未**写入 `pyproject.toml`，主线不依赖它；
- 若要恢复精简环境：`pip uninstall pyqlib` 后重装 `pip install -e ".[dev]"`（pyarrow 需手动回到 26）。

这条记录本身就是 spike 结论的一部分：**qlib 的接入成本不是「装一个库」**。

## 已知的「不是 bug」

| 现象 | 解释 |
|---|---|
| 合成数据上策略跑输买入持有 | `benchmark_warnings` 如实警示；这是正确输出，不是缺陷 |
| 涨停封板导致拒单 | A 股规则建模结果 |
| 组合层与风控上限不一致时抛错 | 装配期失败优于静默全拒 |
| 空 `risk` 字典合法 | 「无拒单」是完整答案 |
